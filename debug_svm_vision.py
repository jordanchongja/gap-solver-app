import cv2
import numpy as np
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
# CHANGE THIS TO YOUR SCREENSHOT PATH
IMAGE_PATH = r"C:\Github Code\gap-solver-app\examples\Screenshot 2026-02-04 115039.png"
GRID_SIZE = 4

# Load the trained model
if not os.path.exists("C:\Github Code\gap-solver-app\model.pkl"):
    print("❌ Error: model.pkl not found. Train your model first!")
    exit()

model_data = joblib.load("C:\Github Code\gap-solver-app\model.pkl")
clf = model_data['model']
classes = model_data['classes']

# ==========================================
# 1. THE EXACT PRE-PROCESSOR (Must match app.py)
# ==========================================
def standardize_cell(img):
    # Resize to 64x64 (Matches your latest training)
    img = cv2.resize(img, (64, 64))
    
    # Blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Corner Difference Strategy
    corner_TL = blurred[0:3, 0:3]
    corner_TR = blurred[0:3, -3:]
    corner_BL = blurred[-3:, 0:3]
    corner_BR = blurred[-3:, -3:]
    corners = np.vstack((corner_TL, corner_TR, corner_BL, corner_BR))
    avg_bg_color = np.mean(corners, axis=(0, 1))
    
    # Difference Calculation
    diff = cv2.absdiff(blurred, avg_bg_color.astype(np.uint8))
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    # Cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # Safety Border
    h, w = mask.shape
    cv2.rectangle(mask, (0,0), (w, h), 0, 2)
    
    # Final Re-Centering (The Seeker)
    coords = cv2.findNonZero(mask)
    final_img = np.zeros((64, 64), dtype=np.uint8)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        shape = mask[y:y+h, x:x+w]
        center_x = (64 - w) // 2
        center_y = (64 - h) // 2
        final_img[center_y:center_y+h, center_x:center_x+w] = shape
        
    return final_img

def smart_crop_board(image, grid_n):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    expected_w = img.shape[1] // grid_n
    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 0.7 < w/h < 1.3 and expected_w*0.2 < w < expected_w*1.5:
            valid_boxes.append((x,y,w,h))
            
    if len(valid_boxes) < 4: return img
    
    min_x = min([b[0] for b in valid_boxes])
    min_y = min([b[1] for b in valid_boxes])
    max_x = max([b[0]+b[2] for b in valid_boxes])
    max_y = max([b[1]+b[3] for b in valid_boxes])
    
    if (max_x - min_x) < img.shape[1] * 0.2: return img
    return img[min_y:max_y, min_x:max_x]

# ==========================================
# 2. DEBUGGER LOOP
# ==========================================
def main():
    if not os.path.exists(IMAGE_PATH):
        print("Image file not found!")
        return

    original_img = cv2.imread(IMAGE_PATH)
    
    # 1. Find Board
    board = smart_crop_board(original_img, GRID_SIZE)
    h, w = board.shape[:2]
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    
    print(f"Board Found. Press SPACE to step through cells. ESC to quit.")

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1, x1 = r*cell_h, c*cell_w
            
            # Slice with margin (Same as App)
            m_h = int(cell_h * 0.05)
            m_w = int(cell_w * 0.05)
            raw_cell = board[y1:y1+cell_h, x1:x1+cell_w]
            safe_cell = raw_cell[m_h:cell_h-m_h, m_w:cell_w-m_w]
            
            # 2. PROCESS (See what the SVM sees)
            svm_input = standardize_cell(safe_cell)
            
            # 3. PREDICT
            feat = svm_input.flatten() / 255.0
            pred_idx = clf.predict([feat])[0]
            # If your model stores string labels directly, pred_idx is the label
            label = str(pred_idx) 
            
            # ==================================
            # VISUALIZATION
            # ==================================
            # Make side-by-side comparison
            # Resize raw cell to 300x300 for easy viewing
            view_raw = cv2.resize(safe_cell, (300, 300))
            
            # Resize SVM input (64x64) up to 300x300 so we can see pixels
            view_svm = cv2.resize(svm_input, (300, 300), interpolation=cv2.INTER_NEAREST)
            view_svm_color = cv2.cvtColor(view_svm, cv2.COLOR_GRAY2BGR)
            
            # Stack them
            combined = np.hstack((view_raw, view_svm_color))
            
            # Draw Info
            # Green text if it detected something, Red if blank
            color = (0, 255, 0) if cv2.countNonZero(svm_input) > 0 else (0, 0, 255)
            
            cv2.rectangle(combined, (0,0), (600, 40), (0,0,0), -1)
            cv2.putText(combined, f"Cell ({r},{c}) -> Prediction: {label}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Debugger: Left=Reality, Right=SVM Brain", combined)
            
            # Wait for key
            key = cv2.waitKey(0)
            if key == 27: # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()