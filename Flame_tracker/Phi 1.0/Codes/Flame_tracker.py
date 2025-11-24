import cv2
import sys
import time
import glob
import os
import numpy as np

# ===================================================================================
# --- MANDATORY VIDEO-SPECIFIC PARAMETERS ---
# ===================================================================================

# --- PRIMARY PARAMETER ---
# If the brightest pixel in the search area is below this, we assume no flame.
BRIGHTNESS_DETECTION_THRESHOLD = 1 

# --- SECONDARY PARAMETERS ---
FRAMES_TO_SEARCH_FOR_BACKGROUND = 20

STATIC_THRESHOLD = 2 
STATIC_ERASER_THRESHOLD = 1
# ===================================================================================


# ===================================================================================
# --- 1. CONFIGURATION ---
# ===================================================================================
# --- File Paths ---
IMAGE_FOLDER_PATH_INPUT = r"D:\FREI_videos_Flame_tracking\Phi_1p0\Phi_1p0_u_0p3_C001H001S0001\Phi_1p0_u_0p3_C001H001S0001_frames"
TRAJECTORY_OUTPUT_FILE = r"D:\FREI_videos_Flame_tracking\Phi_1p0\Phi_1p0_u_0p3_C001H001S0001\Datasets_intensity\Phi_1p0_u_0p3_C001H001S0001.csv"
VIDEO_OUTPUT_FOLDER = r"D:\FREI_videos_Flame_tracking\Phi_1p0\Phi_1p0_u_0p3_C001H001S0001\Recorded tracking videos"

FPS_FOR_TIMESTAMPS = 30.0 

PIXELS_PER_MM = 4.8571  
IGNITION_ZONE_MM = (0.0, 8.6, 210.0, 18.0) 
CLOSE_ZONE_RIGHT_PIPE_MM = (0.0, 9.6, 205.0, 16.6)

# PIPE_ZONE (x1, y1, x2, y2) in mm
PIPE_ZONE = (0.0, 11.0, 210.0, 15.9) 

scale_y_position_mm = 22.0

LOST_TIMEOUT_SECONDS = 0.15
RECORDING_CUTOFF_S = 0.15 
DISPLAY_BOX_SIZE = (45, 30)

SEARCH_AREA_SCALE = 2.0 
MIN_FLAME_AREA_PIXELS = 2.0 
MIN_RECORD_DISTANCE_MM = 0.0
MAX_TIME_INTERVAL_S = 0.0

CONTRAST_FACTOR = 4.0
BINARY_THRESHOLD = 1

FONT = cv2.FONT_HERSHEY_SIMPLEX
# ===================================================================================

def get_centroid_x_in_mm(box, pixels_per_mm, direction='L', total_width_mm=0):
    if box is None: return None
    (x, y, w, h) = box
    center_x_px = x + w / 2
    original_mm = center_x_px / pixels_per_mm
    
    if direction == 'L':
        return (original_mm)
    else: # 'R'
        return (total_width_mm - original_mm)

def initialize_system_from_images(image_folder_path, fps_for_timestamps):
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder_path, ext)))
    
    if not image_paths:
        print(f"[ERROR] No images found in folder: {image_folder_path}")
        sys.exit()
        
    image_paths = sorted(image_paths)
    print(f"--- Found {len(image_paths)} images to process ---")
    
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"[ERROR] Could not read first image: {image_paths[0]}")
        sys.exit()
    
    frame_height, frame_width, _ = first_image.shape
    
    print(f"--- Image Properties: {frame_width}x{frame_height} ---")
    print(f"--- Using DATA FPS for timestamps: {fps_for_timestamps:.2f} ---")
    
    return image_paths, frame_width, frame_height

# --- UPDATED FUNCTION: DARKEST FRAME INSIDE PIPE ZONE ---
def find_best_background_frame(image_paths, search_limit, f_dims, pipe_zone_coords):
    """
    Analyzes the first 'search_limit' frames.
    Finds the frame with the LOWEST TOTAL INTENSITY (Darkest)
    SPECIFICALLY inside the Pipe Zone.
    """
    
    print(f"--- Analyzing first {search_limit} frames to find DARKEST PIPE background ---")
    
    min_total_intensity = float('inf')
    best_frame_index = 0 
    frame_width, frame_height = f_dims
    
    # Unpack Pipe Zone (ensure int)
    px1, py1, px2, py2 = pipe_zone_coords
    
    # Ensure bounds
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(frame_width, px2), min(frame_height, py2)
    
    # Loop up to search_limit
    for i in range(min(search_limit, len(image_paths))):
        
        img_path = image_paths[i]
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue
            
        frame = cv2.resize(frame, (frame_width, frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- CROP TO PIPE ZONE ---
        # We only care if the *pipe* is dark. External noise is ignored.
        roi_pipe = gray[py1:py2, px1:px2]
        
        # --- SUM INTENSITY ---
        # Lower sum = Darker = Empty Pipe
        total_intensity = np.sum(roi_pipe)
        
        if total_intensity < min_total_intensity:
            min_total_intensity = total_intensity
            best_frame_index = i
    
    print(f"--- Selected Frame {best_frame_index} (Darkest Pipe). Score: {min_total_intensity} ---")
    return best_frame_index

def find_initial_flame(frame_gray_for_detection, state, iz_coords):
    """
    Detects flame based purely on brightness/intensity within the Ignition Zone.
    """
    iz_x, iz_y, iz_w, iz_h = iz_coords
    ignition_area_gray = frame_gray_for_detection[iz_y:iz_y + iz_h, iz_x:iz_x + iz_w]
    if ignition_area_gray.size == 0: return

    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(ignition_area_gray)
    
    # Simplified Logic: If pixel is bright enough, it's a flame.
    if max_val >= BRIGHTNESS_DETECTION_THRESHOLD:
        center_x_abs = max_loc[0] + iz_x
        center_y_abs = max_loc[1] + iz_y
        
        db_w, db_h = DISPLAY_BOX_SIZE
        dx = int(center_x_abs - db_w / 2)
        dy = int(center_y_abs - db_h / 2)
        initial_box = (dx, dy, db_w, db_h)
        
        ix, iy, iw, ih = initial_box
        is_STRICTLY_inside = (ix > iz_x and 
                             iy > iz_y and 
                             (ix + iw) < (iz_x + iz_w) and 
                             (iy + ih) < (iz_y + iz_h))
        
        if is_STRICTLY_inside:
            roi_center_x_mm = get_centroid_x_in_mm(initial_box, PIXELS_PER_MM, state['direction'], state['total_width_mm'])
            
            state['flame_id_counter'] += 1
            flame_id = state['flame_id_counter']
            current_frame_num = state['frame_counter']
            initiation_time_s = current_frame_num / state['fps']
            
            print(f"\n[START] Flame f-{flame_id} initiated at Frame {current_frame_num} ({initiation_time_s:.2f}s)")
            print(f"        Initial Position: {roi_center_x_mm:.3f} mm (Brightness: {max_val})\n")
            
            state['flame_data'][flame_id] = {
                'trajectory_points': [(current_frame_num, initiation_time_s, roi_center_x_mm)],
                'end_time_s': None
            }
            
            state['leading_edge_pos_mm'] = roi_center_x_mm 
            state['is_tracking'] = True
            state['lost_since_frame'] = 0
            state['last_known_box'] = [int(v) for v in initial_box]
            state['display_box'] = [int(v) for v in initial_box]

def track_by_max_intensity(frame_gray_for_detection, frame_for_verification, state, f_dims, ignition_zone_pixels):
    """
    Tracks the flame by finding the brightest pixel near the last known location,
    then refining the box using Contours.
    """
    frame_width, frame_height = f_dims
    iz_x, iz_y, iz_w, iz_h = ignition_zone_pixels
    track_success = False

    if state['last_known_box']:
        lx, ly, lw, lh = state['last_known_box']
        sw, sh = int(lw * SEARCH_AREA_SCALE), int(lh * SEARCH_AREA_SCALE)
        sx = max(0, int(lx + lw/2 - sw/2))
        sy = max(0, int(ly + lh/2 - sh/2))
        
        search_area_box = (sx, sy, sw, sh)
        search_area_gray = frame_gray_for_detection[sy:min(sy+sh, frame_height), sx:min(sx+sw, frame_width)]
        
        if search_area_gray.size > 0:
            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(search_area_gray)
            
            if max_val >= BRIGHTNESS_DETECTION_THRESHOLD:
                # Flame Detected via Intensity
                track_success = True

                center_x_abs = max_loc[0] + sx
                center_y_abs = max_loc[1] + sy
                
                db_w, db_h = DISPLAY_BOX_SIZE
                dx = int(center_x_abs - db_w / 2)
                dy = int(center_y_abs - db_h / 2)
                current_box = (dx, dy, db_w, db_h)
                
                # --- CONTOUR REFINEMENT ---
                # Attempt to fit the box tighter to the actual flame shape
                contour_box = list(current_box) 
                c_x, c_y, c_w, c_h = contour_box
                roi_bin = frame_for_verification[c_y:c_y+c_h, c_x:c_x+c_w]
                
                if roi_bin.size > 0:
                    if len(roi_bin.shape) == 3:
                        roi_bin_gray = cv2.cvtColor(roi_bin, cv2.COLOR_BGR2GRAY)
                    else:
                        roi_bin_gray = roi_bin
                        
                    contours, _ = cv2.findContours(roi_bin_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        # Optional: Enforce minimum area size to ignore noise
                        if cv2.contourArea(largest_contour) > MIN_FLAME_AREA_PIXELS:
                            tx, ty, tw, th = cv2.boundingRect(largest_contour)
                            contour_box = [c_x + tx, c_y + ty, tw, th]
                    
                final_roi_box = contour_box
                state['last_known_box'] = final_roi_box
                
                center_x_c1 = final_roi_box[0] + final_roi_box[2] / 2
                center_y_c1 = final_roi_box[1] + final_roi_box[3] / 2
                
                c1_pos_mm = get_centroid_x_in_mm((center_x_c1, center_y_c1, 0, 0), PIXELS_PER_MM, state['direction'], state['total_width_mm'])
                
                if c1_pos_mm is not None:
                    if c1_pos_mm > state['leading_edge_pos_mm']:
                        state['leading_edge_pos_mm'] = c1_pos_mm
                        db_w, db_h = DISPLAY_BOX_SIZE
                        dx_disp = int(center_x_c1 - db_w / 2)
                        dy_disp = int(center_y_c1 - db_h / 2)
                        state['display_box'] = (dx_disp, dy_disp, db_w, db_h)

    # Sanity Check: Ensure tracked object is within valid bounds
    if track_success:
        x, y, w, h = state['last_known_box']
        is_STRICTLY_inside = (x > iz_x and y > iz_y and (x + w) < (iz_x + iz_w) and (y + h) < (iz_y + iz_h))
        if not is_STRICTLY_inside:
            track_success = False
            
    if track_success:
        state['lost_since_frame'] = 0
    else:
        if state['lost_since_frame'] == 0: 
            state['lost_since_frame'] = state['frame_counter']

    if state['is_tracking']:
        if state['display_box']:
            db_x, db_y, db_w, db_h = state['display_box']
            color = (100) if state['lost_since_frame'] > 0 else (255) 
            text = "Lost..." if state['lost_since_frame'] > 0 else f"f-{state['flame_id_counter']} (Tracking)"
            cv2.rectangle(frame_for_verification, (db_x, db_y), (db_x + db_w, db_y + db_h), color, 2)
            cv2.putText(frame_for_verification, text, (db_x, db_y - 10), FONT, 0.6, color, 2)

        seconds_lost = 0
        if state['lost_since_frame'] > 0:
            seconds_lost = (state['frame_counter'] - state['lost_since_frame']) / state['fps']

        is_within_recording_window = (state['lost_since_frame'] == 0) or (seconds_lost <= RECORDING_CUTOFF_S)

        if is_within_recording_window:
            flame_id = state['flame_id_counter']
            last_frame, last_saved_time, last_saved_pos = state['flame_data'][flame_id]['trajectory_points'][-1]
            current_time_s = state['frame_counter'] / state['fps']
            current_pos_for_csv = state['leading_edge_pos_mm']
            distance_moved = abs(current_pos_for_csv - last_saved_pos)
            time_elapsed = current_time_s - last_saved_time
            
            if (distance_moved >= MIN_RECORD_DISTANCE_MM) or (time_elapsed >= MAX_TIME_INTERVAL_S):
                state['flame_data'][flame_id]['trajectory_points'].append((state['frame_counter'], current_time_s, current_pos_for_csv))

        if seconds_lost > LOST_TIMEOUT_SECONDS:
            flame_id = state['flame_id_counter']
            last_pos_mm = state['leading_edge_pos_mm']
            lost_at_frame = state['lost_since_frame']
            current_time_s = (lost_at_frame / state['fps']) + LOST_TIMEOUT_SECONDS
            print(f"\n[END] Flame f-{flame_id} extinguished (Timeout).")
            print(f"        Last seen at Frame {lost_at_frame} ({current_time_s:.2f}s)")
            print(f"        Final Position: {last_pos_mm:.3f} mm\n")
            state['flame_data'][flame_id]['end_time_s'] = current_time_s
            state['is_tracking'] = False

def save_flame_data(state):
    if not state['flame_data']: return
    output_dir = os.path.dirname(TRAJECTORY_OUTPUT_FILE)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(TRAJECTORY_OUTPUT_FILE, 'w', newline='') as f:
            f.write("Frame_ID,Flame_ID,Timestamp_s,Position_MM\n")
            for flame_id, data in sorted(state['flame_data'].items()):
                for frame_id, timestamp, position in data['trajectory_points']:
                    f.write(f"{frame_id},{flame_id},{timestamp:.6f},{position}\n")
        print(f"\nDetailed flame trajectory data saved to '{TRAJECTORY_OUTPUT_FILE}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save data: {e}")

def draw_scale_bar(image, scale_info):
    SCALE_COLOR = (192) 
    pixels_per_mm = scale_info['pixels_per_mm'] 
    y_pos = scale_info['scale_y_position']
    direction = scale_info['direction']
    
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        SCALE_COLOR = 192 
        
    if height == 0 or width == 0: return
    cv2.line(image, (0, y_pos), (width - 1, y_pos), SCALE_COLOR, 1)
    num_ticks = int(width / pixels_per_mm) + 1 
    
    for i in range(num_ticks):
        label_num = i
        current_x = -1
        if direction == 'L':
            current_x = int(i * pixels_per_mm)
        else: # 'R'
            current_x = (width - 1) - int(i * pixels_per_mm)
        
        if current_x < 0 or current_x >= width:
            continue 
        cv2.line(image, (current_x, y_pos - 2), (current_x, y_pos + 2), SCALE_COLOR, 1)
        
        if label_num % 50 == 0: 
            label_text = f"{label_num}"
            (text_width, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            text_x_pos = -1
            if direction == 'L':
                text_x_pos = current_x + 3
            else: # 'R'
                text_x_pos = current_x - text_width - 3 
                if label_num == 0:
                    text_x_pos = current_x - text_width
                if text_x_pos < 0:
                    text_x_pos = 3
            cv2.putText(image, f"{label_text} mm", (text_x_pos, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, SCALE_COLOR, 1)

def main():
    tracking_direction = ''
    while tracking_direction not in ['L', 'R']:
        tracking_direction = input("Enter tracking direction (L for LTR, R for RTL): ").upper()
        if tracking_direction not in ['L', 'R']:
            print("Invalid input. Please enter 'L' or 'R'.")
    
    print(f"Tracking direction set to: {'Left-to-Right' if tracking_direction == 'L' else 'Right-to-Left'}")
    
    # --- CHANGED: Removed template path argument ---
    image_paths, frame_width, frame_height = initialize_system_from_images(
        IMAGE_FOLDER_PATH_INPUT, 
        FPS_FOR_TIMESTAMPS
    )
    
    iz_x_start_px, iz_y_start_px = int(IGNITION_ZONE_MM[0] * PIXELS_PER_MM), int(IGNITION_ZONE_MM[1] * PIXELS_PER_MM)
    iz_x_end_px, iz_y_end_px = int(IGNITION_ZONE_MM[2] * PIXELS_PER_MM), int(IGNITION_ZONE_MM[3] * PIXELS_PER_MM)
    ignition_zone_pixels = (iz_x_start_px, iz_y_start_px, iz_x_end_px - iz_x_start_px, iz_y_end_px - iz_y_start_px)
    
    close_x1 = int(CLOSE_ZONE_RIGHT_PIPE_MM[0] * PIXELS_PER_MM)
    close_y1 = int(CLOSE_ZONE_RIGHT_PIPE_MM[1] * PIXELS_PER_MM)
    close_x2 = int(CLOSE_ZONE_RIGHT_PIPE_MM[2] * PIXELS_PER_MM)
    close_y2 = int(CLOSE_ZONE_RIGHT_PIPE_MM[3] * PIXELS_PER_MM)
    close_region_pixels = (close_x1, close_y1, close_x2, close_y2)
    
    # --- Convert Pipe Zone to pixels ---
    pz_x1, pz_y1 = int(PIPE_ZONE[0] * PIXELS_PER_MM), int(PIPE_ZONE[1] * PIXELS_PER_MM)
    pz_x2, pz_y2 = int(PIPE_ZONE[2] * PIXELS_PER_MM), int(PIPE_ZONE[3] * PIXELS_PER_MM)

    # --- Create the main Region of Interest (ROI) mask ---
    roi_mask = np.zeros((frame_height, frame_width), dtype="uint8")
    cv2.rectangle(roi_mask, (pz_x1, pz_y1), (pz_x2, pz_y2), 255, -1)
    print("--- Created brute-force ROI mask to black out external areas ---")

    scale_y_position_pixels = int(scale_y_position_mm * PIXELS_PER_MM)

    scale_parameters = {
        "pixels_per_mm": PIXELS_PER_MM,
        "scale_y_position": scale_y_position_pixels,
        "direction": tracking_direction,
    }

    os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
    video_basename = os.path.basename(os.path.normpath(IMAGE_FOLDER_PATH_INPUT))
    VIDEO_OUTPUT_PATH = os.path.join(VIDEO_OUTPUT_FOLDER, f"{video_basename}_flame_tracked_intensity.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    total_frames = len(image_paths)
    OUTPUT_VIDEO_FPS = total_frames / 300.0  
    print(f"--- Output video will be saved at {OUTPUT_VIDEO_FPS:.2f} FPS (5 min duration) ---")

    DYNAMIC_PLAYBACK_DELAY = int(1000 / OUTPUT_VIDEO_FPS)
    if DYNAMIC_PLAYBACK_DELAY < 1: 
        DYNAMIC_PLAYBACK_DELAY = 1
    print(f"--- Preview window playback delay set to: {DYNAMIC_PLAYBACK_DELAY} ms ---")

    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, OUTPUT_VIDEO_FPS, (frame_width, frame_height), isColor=True)
    
    print(f"Applying MORPH_CLOSE to region: {CLOSE_ZONE_RIGHT_PIPE_MM}")
    print("Applying a stronger MORPH_OPEN to all other regions.")
    
    # --- NEW: FIND BEST FRAME (Min Intensity inside Pipe Zone) ---
    FRAME_FOR_BACKGROUND = find_best_background_frame(
        image_paths, 
        FRAMES_TO_SEARCH_FOR_BACKGROUND,
        (frame_width, frame_height),
        (pz_x1, pz_y1, pz_x2, pz_y2) # Pass the calculated pixel coords of the pipe
    )

    if FRAME_FOR_BACKGROUND >= len(image_paths):
        print(f"[ERROR] FRAME_FOR_BACKGROUND index ({FRAME_FOR_BACKGROUND}) is out of range.")
        sys.exit()
        
    bg_image_path = image_paths[FRAME_FOR_BACKGROUND]
    bg_frame = cv2.imread(bg_image_path)
    if bg_frame is None:
        print(f"[ERROR] Could not read background frame at: {bg_image_path}")
        sys.exit()
    
    # --- Apply the brute force ROI mask to the background frame ---
    bg_frame = cv2.bitwise_and(bg_frame, bg_frame, mask=roi_mask)
        
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
    
    # --- Create the static "Eraser" mask ---
    _, static_object_mask = cv2.threshold(bg_gray, STATIC_ERASER_THRESHOLD, 255, cv2.THRESH_BINARY)
    print(f"--- Created Static Eraser Mask (Threshold > {STATIC_ERASER_THRESHOLD}) ---")


    close_mask = np.zeros((frame_height, frame_width), dtype="uint8")
    cx1, cy1, cx2, cy2 = close_region_pixels
    cv2.rectangle(close_mask, (cx1, cy1), (cx2, cy2), 255, -1)
    open_mask = cv2.bitwise_not(close_mask)

    close_kernel = np.ones((3, 3), np.uint8)
    close_iterations = 10
    open_kernel = np.ones((20, 20), np.uint8)
    open_iterations = 40

    TOTAL_WIDTH_MM = (frame_width - 1) / PIXELS_PER_MM
    
    state = {
        'is_tracking': False, 'flame_id_counter': 0,
        'last_known_box': None, 'lost_since_frame': 0, 'frame_counter': 0,
        'flame_data': {}, 
        'fps': FPS_FOR_TIMESTAMPS, 
        'display_box': None,
        'leading_edge_pos_mm': 0.0, 
        'direction': tracking_direction,
        'total_width_mm': TOTAL_WIDTH_MM
    }

    print("\nPress 'q' to quit.")
    print("---------------------------------------------------")
    print("--- STARTING FLAME TRACKING (Intensity Only) ---")
    
    # --- Create the "Keeper" mask just once ---
    keeper_mask = cv2.bitwise_not(static_object_mask)
    
    for image_path in image_paths:
        # Skip the specific background frame to avoid self-subtraction artifacts
        if image_path == bg_image_path:
            continue
            
        frame_original = cv2.imread(image_path)
        if frame_original is None:
            print(f"[WARN] Could not read image {image_path}, skipping.")
            continue
            
        # --- Apply the brute force ROI mask to the current frame ---
        frame_original = cv2.bitwise_and(frame_original, frame_original, mask=roi_mask)
            
        state['frame_counter'] += 1
        
        # --- ENTIRE BGS LOGIC ---
        current_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

        diff_mask = cv2.absdiff(bg_gray, current_gray)
        ret, single_fg_mask = cv2.threshold(diff_mask, STATIC_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        fg_in_close_zone = cv2.bitwise_and(single_fg_mask, single_fg_mask, mask=close_mask)
        fg_in_open_zone = cv2.bitwise_and(single_fg_mask, single_fg_mask, mask=open_mask)

        processed_close_zone = cv2.morphologyEx(fg_in_close_zone, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
        processed_open_zone = cv2.morphologyEx(fg_in_open_zone, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)

        fg_mask_cleaned = cv2.bitwise_or(processed_close_zone, processed_open_zone)
        
        flame_region_orig_color = cv2.bitwise_and(frame_original, frame_original, mask=fg_mask_cleaned)
        hsv_flame = cv2.cvtColor(flame_region_orig_color, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_flame)
        v = cv2.convertScaleAbs(v, alpha=CONTRAST_FACTOR, beta=0) 
        merged_hsv = cv2.merge([h, s, v])
        enhanced_flame_bgr = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
        
        # --- 1. This is the "dirty" grayscale image ---
        gray_enhanced_flame = cv2.cvtColor(enhanced_flame_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- 2. Apply the "Keeper" mask to clean the grayscale image ---
        gray_enhanced_flame_CLEAN = cv2.bitwise_and(gray_enhanced_flame, gray_enhanced_flame, mask=keeper_mask)
        
        # --- 3. This is the binarized mask (now created from the CLEAN gray image) ---
        ret, binary_mask = cv2.threshold(gray_enhanced_flame_CLEAN, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # --- 4. This is the FINAL image for drawing and verification ---
        frame_to_process_and_draw = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        iz_x, iz_y, iz_w, iz_h = ignition_zone_pixels
        cv2.rectangle(frame_to_process_and_draw, (iz_x, iz_y), (iz_x + iz_w, iz_y + iz_h), (200, 200, 200), 1)
        
        # --- CHANGED: Removed templates argument ---
        if state['is_tracking']:
            track_by_max_intensity(
                gray_enhanced_flame_CLEAN, 
                frame_to_process_and_draw, 
                state, 
                (frame_width, frame_height), 
                ignition_zone_pixels
            )
        else:
            find_initial_flame(
                gray_enhanced_flame_CLEAN, 
                state, 
                ignition_zone_pixels
            )
        
        current_time_s = state['frame_counter'] / state['fps'] 
        cv2.putText(frame_to_process_and_draw, f"Time: {current_time_s:.2f}s", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        
        draw_scale_bar(frame_to_process_and_draw, scale_parameters)
        
        out.write(frame_to_process_and_draw)
        cv2.imshow("Automated Flame Tracker", frame_to_process_and_draw)
        
        if cv2.waitKey(DYNAMIC_PLAYBACK_DELAY) & 0xFF == ord('q'):
            print("--- User quit processing ---")
            break
            
    # --- 8. CLEANUP ---
    if state['is_tracking'] and state['flame_id_counter'] > 0 and state['flame_data'][state['flame_id_counter']]['end_time_s'] is None:
        flame_id = state['flame_id_counter']
        final_time_s = state['frame_counter'] / state['fps']
        last_pos_mm = state['leading_edge_pos_mm']
        
        print(f"\n[END] Flame f-{flame_id} still tracking at end of video.")
        print(f"        Final Frame: {state['frame_counter']} ({final_time_s:.2f}s)")
        print(f"        Final Position: {last_pos_mm:.3f} mm\n")
        
        state['flame_data'][flame_id]['end_time_s'] = final_time_s

    save_flame_data(state)
    
    out.release()
    cv2.destroyAllWindows()
    
    print("\nProcessing complete.")
    print(f"Tracking video saved to: {VIDEO_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
