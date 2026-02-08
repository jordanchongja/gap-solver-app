import streamlit as st
from streamlit_paste_button import paste_image_button as pbutton
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. CORE LOGIC (Image Processing)
# ==========================================

def crop_to_grid(source_image, grid_n):
    img = source_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cells = []
    img_h, img_w = img.shape[:2]
    min_area = (img_w // 20) ** 2 
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect_ratio = w / float(h)
        
        if len(approx) == 4 and area > min_area and 0.8 < aspect_ratio < 1.2:
            valid_cells.append((x, y, w, h))

    if not valid_cells:
        return source_image

    valid_cells.sort(key=lambda k: (k[1], k[0]))
    expected_cells = grid_n * grid_n
    main_grid_cells = valid_cells[:expected_cells]
    
    if not main_grid_cells:
        return source_image

    min_x = min([c[0] for c in main_grid_cells])
    min_y = min([c[1] for c in main_grid_cells])
    max_x = max([c[0] + c[2] for c in main_grid_cells])
    max_y = max([c[1] + c[3] for c in main_grid_cells])
    
    pad = 5
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(img_w, max_x + pad)
    max_y = min(img_h, max_y + pad)
    
    return source_image[min_y:max_y, min_x:max_x]

def detect_shape_by_contour(cell_image):
    h_full, w_full = cell_image.shape[:2]
    margin_x = int(w_full * 0.15)
    margin_y = int(h_full * 0.15)
    
    roi = cell_image[margin_y:h_full-margin_y, margin_x:w_full-margin_x]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "blank"

    valid_blobs = []
    roi_h, roi_w = roi.shape[:2]
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50: continue 
        _, _, w, h = cv2.boundingRect(c)
        if w > 0.95 * roi_w or h > 0.95 * roi_h: continue
        valid_blobs.append(c)

    if not valid_blobs:
        return "blank"

    if len(valid_blobs) >= 2:
        return "6question"

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
        elif 9 <= vertices <= 10:
            return "5star"
        else:
            if solidity < 0.45: return "6question"
            elif solidity > 0.80: return "4cross"
            else: return "5star"

def process_full_image(pil_image, grid_size):
    open_cv_image = np.array(pil_image.convert('RGB')) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    cropped = crop_to_grid(open_cv_image, grid_size)

    standard_size = 600
    display_img = cv2.resize(cropped, (standard_size, standard_size))
    
    cell_h = standard_size // grid_size
    cell_w = standard_size // grid_size

    detected_grid = []
    
    for r in range(grid_size):
        row_data = []
        for c in range(grid_size):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            m = 6
            cell_roi = display_img[y1+m:y2-m, x1+m:x2-m]
            label = detect_shape_by_contour(cell_roi)
            row_data.append(label)
        detected_grid.append(row_data)
        
    return detected_grid, cropped

# ==========================================
# 2. SOLVER LOGIC (Robust)
# ==========================================

def find_empty(board):
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 'blank':
                return (r, c)
    return None

def solve_with_backtracking(board, all_shapes):
    find = find_empty(board)
    if not find: return True
    row, col = find
    
    row_vals = board[row]
    col_vals = [board[i][col] for i in range(len(board))]
    
    for shape in all_shapes:
        if shape in row_vals or shape in col_vals: continue
        
        board[row][col] = shape
        if solve_with_backtracking(board, all_shapes): return True
        board[row][col] = 'blank'
        
    return False

def run_solver(grid_data, grid_size):
    board_copy = [row[:] for row in grid_data]
    
    # 1. Identify Existing Shapes
    present_shapes = set()
    question_pos = None
    
    for r in range(len(board_copy)):
        for c in range(len(board_copy[0])):
            val = board_copy[r][c]
            if val == '6question':
                question_pos = (r, c)
            elif val != 'blank':
                present_shapes.add(val)
    
    if not question_pos:
        return "No '6question' found!", None
    
    # 2. Force Universe Size == Grid Size
    # If the puzzle is sparse, we might only see 3 shapes in a 4x4 grid.
    # We MUST inject shapes to make the universe length equal to grid_size.
    universe = list(present_shapes)
    
    # Priority list of shapes to add if we are missing some
    standard_shapes = ['1circle', '2triangle', '3square', '4cross', '5star']
    
    for shape in standard_shapes:
        if len(universe) >= grid_size:
            break
        if shape not in universe:
            universe.append(shape)
            
    # Sort for consistency
    universe.sort()
    
    # Debug info for user
    # st.write(f"DEBUG: Solving with universe: {universe}")

    qr, qc = question_pos
    board_copy[qr][qc] = 'blank' 
    
    success = solve_with_backtracking(board_copy, universe)
    
    if success:
        return board_copy[qr][qc], board_copy
    else:
        return "Unsolvable", None

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Gap Solver", layout="wide", page_icon="🧩")
st.title("🧩 Gap Challenge Solver")

# Initialize Session State Variables
if 'paste_id' not in st.session_state: st.session_state.paste_id = 0

with st.sidebar:
    st.header("Settings")
    grid_size = st.radio("Grid Size", (4, 5), index=0)
    st.markdown("---")
    if st.button("Reset / Clear All"):
        st.session_state.clear()
        st.rerun()

col1, col2 = st.columns([1, 1])

# --- COL 1: INPUT ---
with col1:
    st.subheader("1. Input Puzzle")
    # Paste button logic
    paste_result = pbutton("📋 Paste Image", key="paste_btn")
    
    if paste_result.image_data is not None:
        # Check if this is a NEW paste or the same one
        if 'last_image_data' not in st.session_state or \
           st.session_state.last_image_data != paste_result.image_data:
            
            # 1. UPDATE IMAGE STATE
            st.session_state.last_image_data = paste_result.image_data
            
            # 2. INCREMENT PASTE ID (This forces the grid editor to reset)
            st.session_state.paste_id += 1 
            
            # 3. PROCESS IMMEDIATELY
            with st.spinner("Analyzing..."):
                detected_grid, cropped_cv = process_full_image(paste_result.image_data, grid_size)
                st.session_state.grid_data = detected_grid
                st.session_state.cropped_view = cropped_cv
                st.rerun()

        st.image(paste_result.image_data, caption="Current Image", use_column_width=True)
    else:
        st.info("Win+Shift+S (Windows) or Cmd+Shift+4 (Mac) to screenshot.")

# --- COL 2: EDIT & SOLVE ---
with col2:
    st.subheader("2. Verify & Solve")
    
    if 'grid_data' in st.session_state and st.session_state.grid_data is not None:
        st.write("Edit any errors below:")
        
        shape_options = ['blank', '1circle', '2triangle', '3square', '4cross', '5star', '6question']
        
        # We append paste_id to the form key. 
        # When paste_id changes, Streamlit sees this as a BRAND NEW form 
        # and discards the old state, effectively resetting the dropdowns.
        with st.form(key=f"grid_editor_form_{st.session_state.paste_id}"):
            current_rows = st.session_state.grid_data
            updated_grid = []
            
            for r in range(grid_size):
                cols = st.columns(grid_size)
                row_data = []
                for c in range(grid_size):
                    val = current_rows[r][c]
                    if val not in shape_options: val = 'blank'
                    
                    # Key includes paste_id to ensure freshness on new paste
                    user_val = cols[c].selectbox(
                        label="cell",
                        options=shape_options,
                        index=shape_options.index(val),
                        key=f"c_{r}_{c}_{st.session_state.paste_id}",
                        label_visibility="collapsed"
                    )
                    row_data.append(user_val)
                updated_grid.append(row_data)
            
            st.markdown("---")
            solve_btn = st.form_submit_button("✅ Solve Updated Grid", type="primary")

        if solve_btn:
            missing_shape, final_grid = run_solver(updated_grid, grid_size)
            
            if final_grid:
                st.success("Solution Found!")
                clean_name = str(missing_shape).replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").upper()
                st.metric(label="The missing shape is:", value=clean_name)
                
                with st.expander("View Solution Grid"):
                    st.table(final_grid)
            else:
                st.error(f"Status: {missing_shape}")
                st.warning("Check your grid. Ensure you have exactly one '?' and no duplicate shapes in any row/col.")

# --- INSTRUCTIONS ---
st.markdown("---")
with st.expander("ℹ️ Help"):
    st.markdown("""
    **Tips:**
    * **Auto-Reset:** Pasting a new image automatically resets the grid.
    * **Missing Shapes:** If the puzzle is sparse (e.g. only stars and crosses are visible), the solver will assume standard shapes (Circles, Squares) fill the gaps.
    """)