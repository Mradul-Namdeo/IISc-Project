import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

# ==========================================
#        USER CONFIGURATION & SETUP
# ==========================================

# 1. PATHS
BASE_PATH = r'D:\New and more complexed Flame Tracking\1_Mid_S20_Phi_1p15_S_2p5kV_D_300_R3_C001H001S0001'
INPUT_VIDEO_PATH = os.path.join(BASE_PATH, r'Video\1_Mid_S20_Phi_1p15_S_2p5kV_D_300_R3_C001H001S0001_S0001.mp4')

OUTPUT_DIR = os.path.join(BASE_PATH, 'Analysis_Results (Analytical_Derivatives) (1_Mid) (n - (n-1)) (iterations=2)')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'Flame_Tracking_Edge.mp4')
PLOT_8_GRID_PATH = os.path.join(OUTPUT_DIR, 'Analysis_8_Plots_Grid.png')
PLOT_FORMULA_PATH = os.path.join(OUTPUT_DIR, 'Analysis_Formula_Sheet.png')

# 2. PHYSICS
DATA_FPS = 100000.0        
PIXELS_PER_MM = 4.23       

# --- CRITICAL: POLYNOMIAL ORDER ---
# 3 = Cubic Distance -> Parabolic Velocity -> Linear Acceleration
# 4 = Quartic Distance -> Cubic Velocity -> Parabolic Acceleration
POLYNOMIAL_ORDER = 3  

# 3. TRACKING PARAMETERS
Video_stopper = True     
DELAY_SECONDS = 0.5      
FIXED_ROI = (177, 10, 748, 43)
END_LINE_X = 177
RAIL_TOP = FIXED_ROI[1]
RAIL_BOTTOM = FIXED_ROI[1] + FIXED_ROI[3]

# --- SWITCHES ---
ENABLE_MORPH_HEALING = True   
ENABLE_SEVER_SELECT  = True   

# 4. VISUALS
BORDER_THICKNESS = 3     
BORDER_COLOR = (255, 255, 255) 
DISPLAY_SCALE = 0.8      

# 5. IMAGE PROCESSING
APPLY_CLAHE = True      
CLAHE_LIMIT = 2.0       
APPLY_GAMMA = True      
GAMMA_VALUE = 4.0       
K_CLUSTERS = 2
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# KERNELS
k_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

INTENSITY_BAND_THRESHOLD = 150 
START_TRACKING_FRAME = 349
MIN_FLAME_AREA = 15
SEARCH_RADIUS_BFS = 100  

# ==========================================
#           HELPER FUNCTIONS
# ==========================================

def enhance_frame(image_gray):
    enhanced = image_gray.copy()
    if APPLY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)
    if APPLY_GAMMA:
        invGamma = 1.0 / GAMMA_VALUE
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
    return enhanced

def draw_text_outlined(img, text, pos, font, scale, color, thickness):
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 1, lineType=cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, lineType=cv2.LINE_AA)

def find_edge_bfs(mask, y, start_x, max_w, max_radius):
    for r in range(0, max_radius):
        candidates = []
        if r == 0:
            candidates = [start_x]
        else:
            candidates = [start_x + r, start_x - r]
        
        for x in candidates:
            if x <= 0 or x >= max_w - 1:
                continue
            if mask[y, x - 1] == 0 and mask[y, x + 1] == 255:
                return x  
    return None 

# ==========================================
#           MAIN EXECUTION
# ==========================================

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
playback_fps = 30.0 

total_w = (vid_w + 2 * BORDER_THICKNESS) * 2
total_h = (vid_h + 2 * BORDER_THICKNESS) * 3
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, (total_w, total_h))

x_roi, y_roi, w_roi, h_roi = FIXED_ROI
roi_history_buffer = [] 

frame_counter = 0
tracking_active = True
tracking_initialized = False
last_valid_seed_x = 10000 
frames_after_completion = 0
wait_frames = int(playback_fps * DELAY_SECONDS)
current_status = "Waiting"
seed_intensity_val = 0
front_pixel_count_final = 0 

# --- DATA STORAGE ---
data_time_sec = []
data_seed_dist_mm = []  
data_front_dist_mm = [] 

print(f"Processing started...")

while True:
    ret, curr_frame = cap.read()
    if not ret: break
    
    frame_counter += 1
    
    # 1. ROI Extraction
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_roi = curr_gray[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    curr_enhanced_roi = enhance_frame(curr_roi)
    
    # Panels Setup
    panel1_norm = cv2.cvtColor(curr_enhanced_roi, cv2.COLOR_GRAY2BGR)
    panel2_diff = np.zeros_like(panel1_norm)
    panel4_front = np.zeros_like(panel1_norm) 
    panel5_mask = np.zeros_like(panel1_norm)
    panel3_track = curr_frame.copy()
    panel6_data = np.zeros_like(curr_frame)

    if frame_counter >= START_TRACKING_FRAME:
        
        if len(roi_history_buffer) >= 1:
            
            # (n - n-1) logic
            diff_img = cv2.absdiff(roi_history_buffer[0], curr_enhanced_roi)
            diff_img = np.power(diff_img / 255.0, 0.5) * 255.0
            diff_img = diff_img.astype(np.uint8)

            if ENABLE_MORPH_HEALING:
                process_blur = cv2.morphologyEx(diff_img, cv2.MORPH_CLOSE, k_heal, iterations=3)
            else:
                process_blur = diff_img

            panel2_diff = cv2.cvtColor(process_blur, cv2.COLOR_GRAY2BGR)

            pixel_values = process_blur.reshape((-1, 1))
            pixel_values = np.float32(pixel_values)
            
            if len(pixel_values) >= K_CLUSTERS:
                _, labels, centers = cv2.kmeans(pixel_values, K_CLUSTERS, None, CRITERIA, 10, cv2.KMEANS_RANDOM_CENTERS)
                flame_idx = np.argmax(centers)
                labels = labels.flatten()
                
                raw_cluster_mask = (labels == flame_idx).astype(np.uint8) * 255
                raw_cluster_mask = raw_cluster_mask.reshape((h_roi, w_roi))
                
                if ENABLE_SEVER_SELECT:
                    k_cutter = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)) 
                    separated_mask = cv2.erode(raw_cluster_mask, k_cutter, iterations=1)
                    cut_contours, _ = cv2.findContours(separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    filtered_mask = np.zeros_like(raw_cluster_mask)
                    if cut_contours:
                        main_body = max(cut_contours, key=cv2.contourArea)
                        cv2.drawContours(filtered_mask, [main_body], -1, 255, thickness=cv2.FILLED)
                        filtered_mask = cv2.dilate(filtered_mask, k_cutter, iterations=1)
                    raw_cluster_mask = filtered_mask
                
                if tracking_active:
                    current_status = "Active"
                    contours, _ = cv2.findContours(raw_cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_cnt = max(contours, key=cv2.contourArea)
                        
                        if cv2.contourArea(largest_cnt) > MIN_FLAME_AREA:
                            # Seed Logic
                            min_x = tuple(largest_cnt[largest_cnt[:,:,0].argmin()][0])[0]
                            front_edge_points = [pt[0] for pt in largest_cnt if pt[0][0] <= (min_x + 2)]
                            
                            best_seed = None
                            max_val = -1
                            for pt in front_edge_points:
                                val = int(curr_enhanced_roi[pt[1], pt[0]])
                                if val > max_val:
                                    max_val = val
                                    best_seed = pt
                            
                            seed_x_local, seed_y_local = best_seed
                            seed_intensity_val = max_val
                            current_seed_x = x_roi + seed_x_local
                            
                            if current_seed_x <= END_LINE_X:
                                tracking_active = False
                                current_status = "Completed"
                            
                            if tracking_active:
                                lower_bound = max(0, seed_intensity_val - INTENSITY_BAND_THRESHOLD)
                                upper_bound = min(255, seed_intensity_val + INTENSITY_BAND_THRESHOLD)
                                band_mask = cv2.inRange(curr_enhanced_roi, lower_bound, upper_bound)
                                final_mask = cv2.bitwise_and(band_mask, raw_cluster_mask)
                                panel5_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
                                
                                rx, ry, rw, rh = cv2.boundingRect(largest_cnt)
                                global_x_start = x_roi + rx
                                
                                # Trace Front Line
                                front_pixels = []
                                front_pixels.append((seed_x_local, seed_y_local))
                                
                                curr_x_ptr = seed_x_local
                                for y_ptr in range(seed_y_local - 1, -1, -1):
                                    found_x = find_edge_bfs(final_mask, y_ptr, curr_x_ptr, w_roi, SEARCH_RADIUS_BFS)
                                    if found_x is not None: curr_x_ptr = found_x 
                                    front_pixels.append((curr_x_ptr, y_ptr))

                                curr_x_ptr = seed_x_local
                                for y_ptr in range(seed_y_local + 1, h_roi):
                                    found_x = find_edge_bfs(final_mask, y_ptr, curr_x_ptr, w_roi, SEARCH_RADIUS_BFS)
                                    if found_x is not None: curr_x_ptr = found_x
                                    front_pixels.append((curr_x_ptr, y_ptr))

                                front_pixels.sort(key=lambda p: p[1])
                                
                                # --- DATA COLLECTION ---
                                current_time_sec = (frame_counter - START_TRACKING_FRAME) / DATA_FPS
                                data_time_sec.append(current_time_sec)

                                # 1. Seed
                                global_x_seed = x_roi + seed_x_local
                                dist_seed_mm = (vid_w - global_x_seed) / PIXELS_PER_MM
                                data_seed_dist_mm.append(dist_seed_mm)

                                # 2. Average Front
                                if len(front_pixels) > 0:
                                    avg_x_local = np.mean([p[0] for p in front_pixels])
                                else:
                                    avg_x_local = seed_x_local
                                
                                global_x_avg = x_roi + avg_x_local
                                dist_avg_front_mm = (vid_w - global_x_avg) / PIXELS_PER_MM
                                data_front_dist_mm.append(dist_avg_front_mm)
                                # -----------------------

                                # Draw Front
                                true_pixel_count = 0
                                if len(front_pixels) > 0: true_pixel_count = 1
                                if len(front_pixels) > 1:
                                    for i in range(len(front_pixels) - 1):
                                        p1 = front_pixels[i]
                                        p2 = front_pixels[i+1]
                                        true_pixel_count += max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]))
                                        cv2.line(panel4_front, p1, p2, (255, 255, 0), 1)
                                
                                front_pixel_count_final = true_pixel_count
                                cv2.circle(panel4_front, (seed_x_local, seed_y_local), 2, (0, 0, 255), -1)

                                if not tracking_initialized or current_seed_x <= (last_valid_seed_x + 2):
                                    last_valid_seed_x = current_seed_x
                                    tracking_initialized = True
                                    cv2.rectangle(panel3_track, (global_x_start, RAIL_TOP), (global_x_start + rw, RAIL_BOTTOM), (0, 255, 0), 2)
                                    cv2.circle(panel3_track, (current_seed_x, y_roi + seed_y_local), 4, (0, 0, 255), -1)
                else:
                    current_status = "Completed"

    roi_history_buffer.append(curr_enhanced_roi.copy())
    if len(roi_history_buffer) > 2:
        roi_history_buffer.pop(0)

    if Video_stopper and current_status == "Completed":
        frames_after_completion += 1
        if frames_after_completion >= wait_frames:
            break

    # ==========================================
    #       COMPOSITION & VISUALIZATION
    # ==========================================
    fs = max(0.4, vid_h / 450.0) 
    th = max(1, int(vid_h / 250))
    dy = int(vid_h / 6)    

    canvas_p1 = np.zeros_like(curr_frame)
    canvas_p1[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = panel1_norm
    draw_text_outlined(canvas_p1, "P1: CLAHE", (20, int(dy * 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    
    canvas_p2 = np.zeros_like(curr_frame)
    canvas_p2[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = panel2_diff
    
    # Update text based on debug status
    heal_txt = "ON" if ENABLE_MORPH_HEALING else "OFF"
    draw_text_outlined(canvas_p2, f"P2: Temporal Diff", (20, int(dy * 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    
    cv2.line(panel3_track, (END_LINE_X, y_roi), (END_LINE_X, y_roi+h_roi), (0, 0, 255), 2)
    draw_text_outlined(panel3_track, "P3: Tracking Logic", (20, int(dy * 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)

    canvas_p4 = np.zeros_like(curr_frame)
    canvas_p4[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = panel4_front
    draw_text_outlined(canvas_p4, "P4: Flame Front", (20, int(dy * 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 0), th)
    
    canvas_p5 = np.zeros_like(curr_frame)
    canvas_p5[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = panel5_mask
    
    sever_txt = "ON" if ENABLE_SEVER_SELECT else "OFF"
    draw_text_outlined(canvas_p5, f"P5: Binary Mask", (20, int(dy * 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    
    status_color = (0, 255, 0) if current_status == "Active" else (0, 0, 255)
    draw_text_outlined(panel6_data, "P6: Analytics", (20, int(dy * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), th)
    draw_text_outlined(panel6_data, f"Status: {current_status}", (20, int(dy * 2.2)), cv2.FONT_HERSHEY_SIMPLEX, fs, status_color, th)
    
    # Display Physics Time
    phys_t = 0.0
    if frame_counter >= START_TRACKING_FRAME:
        phys_t = (frame_counter - START_TRACKING_FRAME) / DATA_FPS
    
    draw_text_outlined(panel6_data, f"Time: {phys_t:.5f} s", (20, int(dy * 3.5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    draw_text_outlined(panel6_data, f"Front Pixels: {front_pixel_count_final}", (20, int(dy * 4.8)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)

    bd = BORDER_THICKNESS
    bc = BORDER_COLOR
    
    # Grid construction
    row1 = np.hstack((
        cv2.copyMakeBorder(canvas_p1, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=bc),
        cv2.copyMakeBorder(canvas_p2, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=bc)
    ))
    row2 = np.hstack((
        cv2.copyMakeBorder(panel3_track, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=bc),
        cv2.copyMakeBorder(canvas_p4, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=bc)
    ))
    row3 = np.hstack((
        cv2.copyMakeBorder(canvas_p5, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=bc),
        cv2.copyMakeBorder(panel6_data, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=bc)
    ))
    final_grid = np.vstack((row1, row2, row3))
    
    out.write(final_grid)
    cv2.imshow("6-Panel Diagnostic", cv2.resize(final_grid, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
    
    if cv2.waitKey(1) == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

# ==========================================
#         PLOTTING LOGIC (ANALYTICAL)
# ==========================================
print(f"Generating Analytical Derivatives (Order {POLYNOMIAL_ORDER})...")

if len(data_seed_dist_mm) > 5:
    
    # 0. Raw Arrays
    t = np.array(data_time_sec)
    d_seed = np.array(data_seed_dist_mm)
    d_front = np.array(data_front_dist_mm)
    
    # --- A. FIT POLYNOMIALS (The Equation) ---
    # 1. Seed
    coeff_dist_seed = np.polyfit(t, d_seed, POLYNOMIAL_ORDER)
    poly_dist_seed = np.poly1d(coeff_dist_seed)
    poly_vel_seed  = np.polyder(poly_dist_seed, 1) # 1st Derivative
    poly_acc_seed  = np.polyder(poly_dist_seed, 2) # 2nd Derivative

    # 2. Front
    coeff_dist_front = np.polyfit(t, d_front, POLYNOMIAL_ORDER)
    poly_dist_front = np.poly1d(coeff_dist_front)
    poly_vel_front  = np.polyder(poly_dist_front, 1)
    poly_acc_front  = np.polyder(poly_dist_front, 2)

    # --- B. EVALUATE EQUATIONS (Analytical Curves) ---
    # Calculate the Smooth Curve values for every time t
    y_dist_seed_smooth = poly_dist_seed(t)
    y_dist_front_smooth = poly_dist_front(t)
    
    y_vel_seed_smooth = poly_vel_seed(t)
    y_vel_front_smooth = poly_vel_front(t)
    
    y_acc_seed_smooth = poly_acc_seed(t)
    y_acc_front_smooth = poly_acc_front(t)

    # --- C. RAW VELOCITY (For Comparison Only) ---
    v_seed_raw = np.diff(d_seed) / np.diff(t)
    v_front_raw = np.diff(d_front) / np.diff(t)
    t_raw = t[1:]

    # ==========================================
    # IMAGE 1: THE 2x4 GRID
    # ==========================================
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(f"Analytical Flame Tracking (Order {POLYNOMIAL_ORDER})", fontsize=18)
    
    # ROW 1: Distances & Raw Velocities
    # 1. Dist Seed (Data vs Fit)
    axs[0, 0].plot(t, d_seed, 'b.', alpha=0.3, label='Raw')
    axs[0, 0].plot(t, y_dist_seed_smooth, 'b-', label='Poly Fit')
    axs[0, 0].set_title("1) Dist: Seed (Fit)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Dist Front (Data vs Fit)
    axs[0, 1].plot(t, d_front, 'g.', alpha=0.3, label='Raw')
    axs[0, 1].plot(t, y_dist_front_smooth, 'g-', label='Poly Fit')
    axs[0, 1].set_title("2) Dist: Front (Fit)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Raw Vel Seed
    axs[0, 2].plot(t_raw, v_seed_raw, 'r-', alpha=0.5)
    axs[0, 2].set_title("3) Vel: Seed (Discrete Raw)")
    axs[0, 2].grid(True)

    # 4. Raw Vel Front
    axs[0, 3].plot(t_raw, v_front_raw, 'orange', alpha=0.5)
    axs[0, 3].set_title("4) Vel: Front (Discrete Raw)")
    axs[0, 3].grid(True)

    # ROW 2: ANALYTICAL DERIVATIVES (The Smooth Equations)
    
    # 5. Vel Seed (Derivative)
    axs[1, 0].plot(t, y_vel_seed_smooth, 'b-', linewidth=2)
    axs[1, 0].set_title("5) Vel: Seed (Analytical dD/dt)")
    axs[1, 0].set_ylabel("Vel (mm/s)")
    axs[1, 0].grid(True)

    # 6. Vel Front (Derivative)
    axs[1, 1].plot(t, y_vel_front_smooth, 'g-', linewidth=2)
    axs[1, 1].set_title("6) Vel: Front (Analytical dD/dt)")
    axs[1, 1].grid(True)

    # 7. Acc Seed (2nd Derivative)
    axs[1, 2].plot(t, y_acc_seed_smooth, 'purple', linewidth=2)
    axs[1, 2].set_title("7) Acc: Seed (Analytical d²D/dt²)")
    axs[1, 2].set_ylabel("Acc (mm/s²)")
    axs[1, 2].grid(True)

    # 8. Acc Front (2nd Derivative)
    axs[1, 3].plot(t, y_acc_front_smooth, 'brown', linewidth=2)
    axs[1, 3].set_title("8) Acc: Front (Analytical d²D/dt²)")
    axs[1, 3].grid(True)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(PLOT_8_GRID_PATH)
    print(f"Saved 8-Plot Grid: {PLOT_8_GRID_PATH}")
    plt.close()

    # ==========================================
    # IMAGE 2: FORMULAS
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(12, 12))
    ax2.axis('off')
    
    # Get Equation Strings
    # 1. Seed Equations
    d_seed_str = str(poly_dist_seed).replace('\n', '')
    v_seed_str = str(poly_vel_seed).replace('\n', '')
    a_seed_str = str(poly_acc_seed).replace('\n', '')
    
    # 2. Front Equations
    d_front_str = str(poly_dist_front).replace('\n', '')
    v_front_str = str(poly_vel_front).replace('\n', '')
    a_front_str = str(poly_acc_front).replace('\n', '')

    formula_text = (
        f"ANALYTICAL FLAME ANALYSIS SHEET\n"
        f"===============================\n\n"
        f"METHODOLOGY:\n"
        f"1. Fit polynomial D(t) of Order {POLYNOMIAL_ORDER} to raw Distance data.\n"
        f"2. Velocity V(t) = Analytical Derivative D'(t).\n"
        f"3. Acceleration A(t) = Analytical Derivative D''(t).\n\n"
        
        f"--------------------------------------------------\n"
        f"SET A: SEED POINT EQUATIONS\n"
        f"--------------------------------------------------\n"
        f"1. Distance D(t):\n"
        f"   {d_seed_str}\n\n"
        f"2. Velocity V(t) [Plot 5]:\n"
        f"   {v_seed_str}\n\n"
        f"3. Acceleration A(t) [Plot 7]:\n"
        f"   {a_seed_str}\n\n\n"

        f"--------------------------------------------------\n"
        f"SET B: FLAME FRONT EQUATIONS\n"
        f"--------------------------------------------------\n"
        f"1. Distance D(t):\n"
        f"   {d_front_str}\n\n"
        f"2. Velocity V(t) [Plot 6]:\n"
        f"   {v_front_str}\n\n"
        f"3. Acceleration A(t) [Plot 8]:\n"
        f"   {a_front_str}\n"
    )
    
    ax2.text(0.05, 0.95, formula_text, fontsize=11, family='monospace', va='top')
    plt.tight_layout()
    plt.savefig(PLOT_FORMULA_PATH)
    print(f"Saved Formula Sheet: {PLOT_FORMULA_PATH}")

else:
    print("Not enough data collected to generate plots.")
