import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import glob

# ==========================================
# 0. SETUP FOLDERS
# ==========================================
BASE_DIR = "dataset"
LABELS = ['1circle', '2triangle', '3square', '4cross', '5star', '6question', 'blank']

# Create directories if they don't exist
os.makedirs(os.path.join(BASE_DIR, "unsorted"), exist_ok=True)
for l in LABELS:
    os.makedirs(os.path.join(BASE_DIR, l), exist_ok=True)

# ==========================================
# 1. SMART CROPPER (Finds the Board)
# ==========================================
def smart_crop_board(image, grid_n):
    """
    Detects the puzzle board within a screenshot by finding 
    clusters of 'cell-like' squares and cropping to their collective boundary.
    """
    # 1. Standardize
    img = image.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Thresholding (The "Dark Mode" Fix)
    # Unlike Canny, this looks for local contrast differences. 
    # It finds faint grey lines on black backgrounds effectively.
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )

    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Filter for "Cell-sized" Candidates
    # We estimate what a cell *should* look like size-wise
    expected_cell_w = w // grid_n
    expected_cell_h = h // grid_n
    min_dim = expected_cell_w * 0.2  # Allow somewhat smaller (partial detection)
    max_dim = expected_cell_w * 1.5  # Allow somewhat larger (thick borders)
    
    valid_boxes = []
    
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch)
        
        # Check 1: Is it roughly square? (0.7 to 1.3 aspect ratio)
        # Check 2: Is it within the expected size range?
        if 0.7 < aspect < 1.3 and min_dim < cw < max_dim and min_dim < ch < max_dim:
            valid_boxes.append((x, y, cw, ch))
            
    # 5. Fallback: If we didn't find enough cells, return original
    # (Maybe the user cropped it perfectly already, or it's a weird image)
    if len(valid_boxes) < 4: 
        return img

    # 6. Calculate the "Cluster" Bounding Box
    # We find the min/max coordinates of ALL valid cells found.
    # This effectively "snaps" to the outer grid lines.
    min_x = min([b[0] for b in valid_boxes])
    min_y = min([b[1] for b in valid_boxes])
    max_x = max([b[0] + b[2] for b in valid_boxes])
    max_y = max([b[1] + b[3] for b in valid_boxes])
    
    # 7. Sanity Check
    # If the cropped area is tiny (e.g. noise), ignore it.
    if (max_x - min_x) < w * 0.3 or (max_y - min_y) < h * 0.3:
        return img
        
    # 8. Return the Crop
    return img[min_y:max_y, min_x:max_x]

# ==========================================
# 2. HELPER: SHAPE GUESSER (For Auto-Sorting)
# ==========================================
def detect_shape_by_contour(cell_image):
    # This is your old logic, just used to guess the folder name.
    # It doesn't need to be perfect, just "good enough" to save sorting time.
    h_full, w_full = cell_image.shape[:2]
    margin_x = int(w_full * 0.15)
    margin_y = int(h_full * 0.15)
    
    roi = cell_image[margin_y:h_full-margin_y, margin_x:w_full-margin_x]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return "blank"

    valid_blobs = []
    roi_h, roi_w = roi.shape[:2]
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50: continue 
        _, _, w, h = cv2.boundingRect(c)
        if w > 0.95 * roi_w or h > 0.95 * roi_h: continue
        valid_blobs.append(c)

    if not valid_blobs: return "blank"
    if len(valid_blobs) >= 2: return "6question"

    largest_contour = max(valid_blobs, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    perimeter = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.035 * perimeter, True) 
    vertices = len(approx)
    is_convex = cv2.isContourConvex(approx)
    _, _, w, h = cv2.boundingRect(largest_contour)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0

    if is_convex or solidity > 0.92:
        if extent < 0.75:
            if solidity > 0.9: return "2triangle"
            else: return "4cross"
        if vertices == 4: return "3square"
        elif vertices == 3: return "2triangle"
        else: return "1circle"
    else:
        if vertices >= 11:
            if solidity > 0.55: return "4cross"
            else: return "6question"
        elif 9 <= vertices <= 10: return "5star"
        else:
            if solidity < 0.45: return "6question"
            elif solidity > 0.80: return "4cross"
            else: return "5star"

# ==========================================
# 3. MAIN PROCESSING PIPELINE
# ==========================================

def process_single_image(cv_image, grid_size, origin_name="img"):
    # STEP 1: Smart Crop (Finds the board inside the screenshot)
    board_img = smart_crop_board(cv_image, grid_size)
    
    # STEP 2: Blind Slice (Now safe to do, because board_img is tight)
    img_h, img_w = board_img.shape[:2]
    cell_h = img_h // grid_size
    cell_w = img_w // grid_size
    
    saved_count = 0
    timestamp = int(time.time() * 1000)

    # STEP 3: Iterate & Save
    for r in range(grid_size):
        for c in range(grid_size):
            y1 = r * cell_h
            x1 = c * cell_w
            
            # Extract basic cell
            cell = board_img[y1:y1+cell_h, x1:x1+cell_w]
            
            # Apply Safety Margin (The "15% Rule")
            # We cut off the edges to ensure no grid lines remain
            margin_h = int(cell_h * 0.15)
            margin_w = int(cell_w * 0.15)
            safe_cell = cell[margin_h : cell_h - margin_h, margin_w : cell_w - margin_w]
            
            # Auto-Sort Guess logic
            try:
                guessed_label = detect_shape_by_contour(safe_cell) 
            except:
                guessed_label = "unsorted"
            
            if guessed_label not in LABELS: guessed_label = "unsorted"
            
            # Filename logic
            safe_origin = origin_name.split('.')[0][-10:]
            filename = f"{guessed_label}_{timestamp}_{safe_origin}_{r}_{c}.png"
            save_path = os.path.join(BASE_DIR, guessed_label, filename)
            
            cv2.imwrite(save_path, safe_cell)
            saved_count += 1
            
    return saved_count

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Batch Data Processor", layout="wide", page_icon="🏭")
st.title("🏭 Batch Data Processor v2")
st.markdown("Robust cropping enabled: Supports Dark Mode & messy screenshots.")

with st.sidebar:
    st.header("Settings")
    grid_size = st.radio("Grid Size for Batch", (4, 5), index=0)
    st.info("⚠️ Make sure all images in your batch match this grid size!")

# TABS
tab1, tab2 = st.tabs(["📂 Process Folder (Recommended)", "⬆️ Upload Files"])

# --- TAB 1: LOCAL FOLDER ---
with tab1:
    st.write("Point to a folder on your computer containing screenshots.")
    
    # Defaults to a folder named 'raw_images' in the current directory
    default_path = os.path.join(os.getcwd(), "raw_images")
    folder_path = st.text_input("Folder Path:", value=default_path)
    
    if st.button("Start Folder Batch", type="primary"):
        if not os.path.exists(folder_path):
            st.error("Folder not found! Please check the path.")
        else:
            # Find images
            extensions = ['*.png', '*.jpg', '*.jpeg']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            if not files:
                st.warning("No images found in that folder.")
            else:
                st.write(f"Found {len(files)} images. Processing...")
                
                # Progress Bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_cells = 0
                
                for i, file_path in enumerate(files):
                    img = cv2.imread(file_path)
                    if img is not None:
                        f_name = os.path.basename(file_path)
                        count = process_single_image(img, grid_size, f_name)
                        total_cells += count
                    
                    # Update Progress
                    progress = (i + 1) / len(files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i+1}/{len(files)}: {os.path.basename(file_path)}")
                
                st.success(f"🎉 Batch Complete! Extracted {total_cells} cells into '{BASE_DIR}'.")
                st.balloons()

# --- TAB 2: UPLOAD FILES ---
with tab2:
    st.write("Drag and drop multiple files here.")
    uploaded_files = st.file_uploader("Upload Screenshots", accept_multiple_files=True, type=['png', 'jpg'])
    
    if uploaded_files and st.button("Process Uploaded Files"):
        progress_bar = st.progress(0)
        total_cells = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            count = process_single_image(img, grid_size, uploaded_file.name)
            total_cells += count
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        st.success(f"Done! Saved {total_cells} cells.")

# --- STATS ---
st.markdown("---")
st.subheader("Dataset Statistics")
cols = st.columns(len(LABELS))
for i, l in enumerate(LABELS):
    path = os.path.join(BASE_DIR, l)
    count = len(os.listdir(path))
    cols[i].metric(l, count)