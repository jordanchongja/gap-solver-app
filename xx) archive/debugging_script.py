import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# REPLACE THIS WITH YOUR FILE PATH
IMAGE_PATH = r"C:\Github Code\gap-solver-app\examples\xx) done\4\Screenshot 2026-02-03 151046.png"
GRID_SIZE = 4

# ==========================================
# 1. SMART CROPPER (Macro - Finds the Board)
# ==========================================
def smart_crop_board_with_coords(image, grid_n):
    img = image.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    expected_cell_w = w // grid_n
    min_dim = expected_cell_w * 0.2
    max_dim = expected_cell_w * 1.5
    
    valid_boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch)
        if 0.7 < aspect < 1.3 and min_dim < cw < max_dim and min_dim < ch < max_dim:
            valid_boxes.append((x, y, cw, ch))
            
    if len(valid_boxes) < 4: 
        return img, (0, 0)

    min_x = min([b[0] for b in valid_boxes])
    min_y = min([b[1] for b in valid_boxes])
    max_x = max([b[0] + b[2] for b in valid_boxes])
    max_y = max([b[1] + b[3] for b in valid_boxes])
    
    if (max_x - min_x) < w * 0.3 or (max_y - min_y) < h * 0.3:
        return img, (0, 0)
        
    cropped_img = img[min_y:max_y, min_x:max_x]
    offset = (min_x, min_y)
    
    return cropped_img, offset

# ==========================================
# 2. MICRO CROPPER (The "Seeker" Step)
# ==========================================
def get_centered_roi(cell_image):
    """
    Finds the shape inside the cell and returns a crop centered on it.
    If no shape is found, returns a default center crop.
    Returns: (cropped_img, (x, y, w, h) relative to cell)
    """
    h, w = cell_image.shape[:2]
    
    # 1. Quick Threshold to find blobs
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for significant blobs (ignore noise)
    valid_blobs = [c for c in contours if cv2.contourArea(c) > 50]
    
    # Standard "Safe Zone" Size (70% of cell)
    target_w = int(w * 0.70)
    target_h = int(h * 0.70)
    
    # A. If we found a shape, center on it!
    if valid_blobs:
        # Find the combined bounding box of all blobs (handles '?' which has 2 parts)
        min_bx = w
        min_by = h
        max_bx = 0
        max_by = 0
        
        for c in valid_blobs:
            bx, by, bw, bh = cv2.boundingRect(c)
            min_bx = min(min_bx, bx)
            min_by = min(min_by, by)
            max_bx = max(max_bx, bx + bw)
            max_by = max(max_by, by + bh)
            
        # Center of the shape
        center_x = (min_bx + max_bx) // 2
        center_y = (min_by + max_by) // 2
        
        # Calculate new crop coordinates centered on that point
        x1 = max(0, center_x - target_w // 2)
        y1 = max(0, center_y - target_h // 2)
        x2 = min(w, x1 + target_w)
        y2 = min(h, y1 + target_h)
        
        # Adjust if we hit the edges
        if x2 - x1 < target_w: x1 = max(0, x2 - target_w)
        if y2 - y1 < target_h: y1 = max(0, y2 - target_h)
        
        return cell_image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

    # B. If empty, just return the default center (Blind Crop)
    else:
        margin_x = (w - target_w) // 2
        margin_y = (h - target_h) // 2
        return cell_image[margin_y:margin_y+target_h, margin_x:margin_x+target_w], \
               (margin_x, margin_y, target_w, target_h)

# ==========================================
# 3. SHAPE RECOGNITION (Unchanged)
# ==========================================
def detect_shape_by_contour(cell_image):
    # This logic now receives a perfectly centered image, 
    # so we don't need complex internal cropping anymore.
    roi = cell_image.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return "blank", "Empty", thresh

    valid_blobs = []
    roi_h, roi_w = roi.shape[:2]
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50: continue 
        x, y, w, h = cv2.boundingRect(c)
        if w > 0.99 * roi_w or h > 0.99 * roi_h: continue
        valid_blobs.append(c)

    if not valid_blobs: return "blank", "Noise/Empty", thresh
    if len(valid_blobs) >= 2: return "6question", f"Blobs: {len(valid_blobs)}", thresh

    largest_contour = max(valid_blobs, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    perimeter = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.035 * perimeter, True) 
    vertices = len(approx)
    is_convex = cv2.isContourConvex(approx)
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0

    debug_info = f"V:{vertices} Ext:{extent:.2f} Sol:{solidity:.2f}"

    if is_convex or solidity > 0.92:
        if extent < 0.75:
            if solidity > 0.9: return "2triangle", debug_info, thresh
            else: return "4cross", debug_info, thresh
        if vertices == 4: return "3square", debug_info, thresh
        elif vertices == 3: return "2triangle", debug_info, thresh
        else: return "1circle", debug_info, thresh
    else:
        if vertices >= 11:
            if solidity > 0.55: return "4cross", debug_info, thresh
            else: return "6question", debug_info, thresh
        elif 9 <= vertices <= 10: return "5star", debug_info, thresh
        else:
            if solidity < 0.45: return "6question", debug_info, thresh
            elif solidity > 0.80: return "4cross", debug_info, thresh
            else: return "5star", debug_info, thresh

# ==========================================
# 4. VISUALIZATION LOOP (Updated)
# ==========================================
def main():
    if not os.path.exists(IMAGE_PATH): 
        print(f"Error: File not found at {IMAGE_PATH}")
        return

    original_img = cv2.imread(IMAGE_PATH)
    
    # 1. Run Smart Cropper
    board_img, (offset_x, offset_y) = smart_crop_board_with_coords(original_img, GRID_SIZE)
    
    board_h, board_w = board_img.shape[:2]
    cell_h = board_h // GRID_SIZE
    cell_w = board_w // GRID_SIZE
    
    print(f"Board Detected. Press SPACE for next cell, 'q' to quit.")

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            
            # --- 1. Coordinate Math ---
            crop_y1 = r * cell_h
            crop_x1 = c * cell_w
            crop_y2 = crop_y1 + cell_h
            crop_x2 = crop_x1 + cell_w
            
            orig_y1 = offset_y + crop_y1
            orig_x1 = offset_x + crop_x1
            orig_y2 = offset_y + crop_y2
            orig_x2 = offset_x + crop_x2

            # --- 2. Extract RAW Cell ---
            raw_cell = board_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # --- 3. MICRO-CROP (The Seeker) ---
            # Finds the shape and re-centers the crop
            centered_cell, (roi_x, roi_y, roi_w, roi_h) = get_centered_roi(raw_cell)
            
            # Run Detection on the CLEAN, CENTERED image
            label, stats, debug_mask = detect_shape_by_contour(centered_cell)

            # --- 4. Visualization Construction ---
            
            # A. ZOOM WINDOW
            view_size = 300
            if centered_cell.size == 0: continue
            
            roi_zoom = cv2.resize(centered_cell, (view_size, view_size))
            mask_zoom = cv2.resize(debug_mask, (view_size, view_size))
            mask_zoom_bgr = cv2.cvtColor(mask_zoom, cv2.COLOR_GRAY2BGR)
            
            combined_zoom = np.hstack((roi_zoom, mask_zoom_bgr))
            cv2.putText(combined_zoom, f"Pos: ({r},{c})", (10, 30), 1, 1.5, (0, 255, 0), 2)
            cv2.putText(combined_zoom, f"Pred: {label}", (10, 70), 1, 1.5, (0, 0, 255), 2)
            cv2.putText(combined_zoom, stats, (10, view_size - 10), 1, 1.0, (255, 255, 0), 1)

            # B. CONTEXT WINDOW
            full_view = original_img.copy()
            
            # Blue Box = Board
            cv2.rectangle(full_view, (offset_x, offset_y), (offset_x + board_w, offset_y + board_h), (255, 0, 0), 2)
            
            # Red Box = Raw Cell Grid
            cv2.rectangle(full_view, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 0, 255), 4)
            
            # Green Box = The "Seeker" Result (Centered on Shape)
            # Need to map ROI coordinates back to full image
            green_x1 = orig_x1 + roi_x
            green_y1 = orig_y1 + roi_y
            green_x2 = green_x1 + roi_w
            green_y2 = green_y1 + roi_h
            
            cv2.rectangle(full_view, (green_x1, green_y1), (green_x2, green_y2), (0, 255, 0), 2)
            
            # Resize full view
            disp_h = 700
            scale = disp_h / original_img.shape[0]
            disp_w = int(original_img.shape[1] * scale)
            full_view_resized = cv2.resize(full_view, (disp_w, disp_h))

            # --- 5. Show Windows ---
            cv2.imshow("1. Full Context (Green=Focused Shape)", full_view_resized)
            cv2.imshow("2. Analysis (Centered)", combined_zoom)
            
            cv2.moveWindow("1. Full Context (Green=Focused Shape)", 0, 0)
            cv2.moveWindow("2. Analysis (Centered)", disp_w + 20, 0)

            key = cv2.waitKey(0)
            if key == ord('q'): 
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Debugging Complete.")

if __name__ == "__main__":
    main()