import cv2
import sys
import time
import glob
import os
import numpy as np

# ===================================================================================
# --- 1. CONFIGURATION ---
# ===================================================================================
# --- MODIFIED: File Paths ---
IMAGE_FOLDER_PATH_INPUT = r"D:\FREI_videos_Flame_tracking\Phi_1p0\Phi_1p0_u_0p2_C001H001S0001\Phi_1p0_u_0p2_C001H001S0001_frames"
TEMPLATE_FOLDER_PATH = r"D:\FREI_videos_Flame_tracking\Phi_1p2\Phi_1p2_u_0p4_C001H001S0001\Phi_1p0_u_0p3_C001H001S0001_similar_images"
# --- MODIFIED: Output CSV path updated ---
TRAJECTORY_OUTPUT_FILE = r"D:\FREI_videos_Flame_tracking\Phi_1p2\Phi_1p2_u_0p4_C001H001S0001\NEWDatasets\Phi_1p2_u_0p4_C001H001S0001.csv"
VIDEO_OUTPUT_FOLDER = r"D:\FREI_videos_Flame_tracking\Phi_1p2\Phi_1p2_u_0p4_C001H001S0001\Recorded tracking videos"

# --- NEW: FPS for Timestamps (Data Clock) ---
FPS_FOR_TIMESTAMPS = 30.0 

# --- MODIFIED: Physical Setup (in mm) ---
PIXELS_PER_MM = 4.8571  # Was 48.571 (PIXELS_PER_CM / 10)
# This is for the *tracker* (where to look for new flames)
IGNITION_ZONE_MM = (0.0, 8.6, 210.0, 17.7) # Values * 10
# This is for the *background subtractor* (region for sensitive subtraction)
CLOSE_ZONE_RIGHT_PIPE_MM = (0.0, 9.6, 202.0, 16.6) # Values * 10, updated 20.0 -> 200.0
# This is for the scale bar
scale_y_position_mm = 22.0 # Value * 10

# --- Tracking Parameters ---
CONFIDENCE_THRESHOLD = 0.3
EXTINGUISH_THRESHOLD = -1.5
LOST_TIMEOUT_SECONDS = 0.10
RECORDING_CUTOFF_S = 0.10 
DISPLAY_BOX_SIZE = (45, 30)

# --- Advanced Refinement ---
USE_PERIODIC_REDETECTION = True
REDETECTION_INTERVAL = 2      # 8 :)
SEARCH_AREA_SCALE = 2.0
MIN_FLAME_AREA_PIXELS = 0
MIN_RECORD_DISTANCE_MM = 0.0 # Was 0.001 (MIN_RECORD_DISTANCE_CM * 10)
MAX_TIME_INTERVAL_S = 0.0

# --- MODIFIED: Background Subtraction Parameters ---
# --- CONTROL KNOB 1: Contrast ---
CONTRAST_FACTOR = 1.25
# --- CONTROL KNOB 2: Binarization Threshold ---
BINARY_THRESHOLD = 7
# --- NEW: Static Background Subtraction Controls ---
# Define which frame (by index) from the image folder to use as the background
FRAME_FOR_BACKGROUND = 1 
# This is the new sensitivity knob (Lower = more sensitive)
STATIC_THRESHOLD = 2 
# --- REMOVED: MOG2 Parameters ---

# --- Display Settings ---
# --- REMOVED: PLAYBACK_SPEED_DELAY = 270 (Will be calculated dynamically) ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ===================================================================================

# --- MODIFIED: Renamed function and variables to _mm ---
def get_centroid_x_in_mm(box, pixels_per_mm, direction='L', total_width_mm=0):
    """
    Calculates the x-coordinate of the centroid in mm,
    conditionally inverting it for RTL tracking.
    """
    if box is None: return None
    (x, y, w, h) = box
    center_x_px = x + w / 2
    
    original_mm = center_x_px / pixels_per_mm
    
    if direction == 'L':
        return (original_mm)
    else: # 'R'
        # Invert the coordinate system: 0 is on the right
        return (total_width_mm - original_mm)

def re_detect_flame(frame, templates, search_area_box):
    """
    Performs template matching within a specific search area to re-detect a lost flame.
    """
    sx, sy, sw, sh = search_area_box
    frame_height, frame_width, _ = frame.shape
    search_area = frame[sy:min(sy+sh, frame_height), sx:min(sx+sw, frame_width)]
    if search_area.size == 0: return None
    
    # --- MODIFIED: Frame is binarized, use grayscale channel ---
    if len(search_area.shape) == 3:
        search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
    else:
        search_gray = search_area
        
    best_score = -1
    best_roi_rel = None
    best_template_dims = (0, 0)
    for t in templates:
        th, tw = t.shape
        if not (search_gray.shape[0] < th or search_gray.shape[1] < tw):
            res = cv2.matchTemplate(search_gray, t, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)
            if val > best_score:
                best_score = val
                best_roi_rel = loc
                best_template_dims = (tw, th)
    if best_score >= CONFIDENCE_THRESHOLD:
        rx, ry = best_roi_rel
        rw, rh = best_template_dims
        return (rx + sx, ry + sy, rw, rh)
    return None

# --- MODIFIED: This function now initializes from an image folder ---
def initialize_system_from_images(image_folder_path, template_folder_path, fps_for_timestamps):
    """Initializes by loading images, templates, and retrieves image properties."""
    
    # --- 1. Load Image Paths ---
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder_path, ext)))
    
    if not image_paths:
        print(f"[ERROR] No images found in folder: {image_folder_path}")
        sys.exit()
        
    # --- CRITICAL: Sort images to ensure correct order ---
    image_paths = sorted(image_paths)
    print(f"--- Found {len(image_paths)} images to process ---")
    
    # --- 2. Get Frame Dimensions from first image ---
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"[ERROR] Could not read first image: {image_paths[0]}")
        sys.exit()
    
    frame_height, frame_width, _ = first_image.shape
    
    # --- 3. Load Templates ---
    template_extensions = ('*.png', '*.jpg', '*.jpeg')
    template_paths = []
    for ext in template_extensions:
        template_paths.extend(glob.glob(os.path.join(template_folder_path, ext)))
    
    templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in template_paths]
    templates = [t for t in templates if t is not None]
    
    if not templates:
        print(f"[ERROR] No valid template images found in '{template_folder_path}'. Exiting.")
        sys.exit()
    
    print(f"--- Flame Tracker: Loaded {len(templates)} templates from folder ---")
    print(f"--- Image Properties: {frame_width}x{frame_height} ---")
    print(f"--- Using DATA FPS for timestamps: {fps_for_timestamps:.2f} ---")
    
    return image_paths, templates, frame_width, frame_height, fps_for_timestamps

def process_detection_mode(frame, state, templates, iz_coords):
    """Searches for a new flame in the ignition zone and initiates tracking if found."""
    iz_x, iz_y, iz_w, iz_h = iz_coords
    ignition_area = frame[iz_y:iz_y + iz_h, iz_x:iz_x + iz_w]
    # --- MODIFIED: Frame is already binarized, so use grayscale channel ---
    if len(frame.shape) == 3:
        ignition_area_gray = cv2.cvtColor(ignition_area, cv2.COLOR_BGR2GRAY)
    else:
        ignition_area_gray = ignition_area
        
    best_match_score = -1
    best_match_roi = None
    for template in templates:
        t_h, t_w = template.shape
        if not (ignition_area_gray.shape[0] < t_h or ignition_area_gray.shape[1] < t_w):
            result = cv2.matchTemplate(ignition_area_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_match_score:
                best_match_score = max_val
                top_left_abs = (max_loc[0] + iz_x, max_loc[1] + iz_y)
                best_match_roi = (top_left_abs[0], top_left_abs[1], t_w, t_h)

    if best_match_score >= CONFIDENCE_THRESHOLD:
        x, y, w, h = best_match_roi
        
        # --- NEW: Perform contour analysis to check size before starting ---
        roi = frame[y:y+h, x:x+w]
        actual_area = 0
        if roi.size > 0:
            # Video is already B/W, so just convert to find contours
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
            contours, _ = cv2.findContours(gray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                actual_area = cv2.contourArea(largest_contour)

        # --- NEW: Size check to prevent tracking noise ---
        if actual_area >= state['min_flame_area_check']:
            # This is a valid flame, not just noise. Start tracking.
            center_x = x + w / 2
            center_y = y + h / 2
            
            db_w, db_h = DISPLAY_BOX_SIZE
            dx = int(center_x - db_w / 2)
            dy = int(center_y - db_h / 2)
            display_box = (dx, dy, db_w, db_h)
            
            # --- MODIFIED: Using _mm variables ---
            roi_center_x_mm = get_centroid_x_in_mm(display_box, PIXELS_PER_MM, state['direction'], state['total_width_mm'])
            
            state['flame_id_counter'] += 1
            flame_id = state['flame_id_counter']
            current_frame_num = state['frame_counter'] # Get frame num for CSV
            initiation_time_s = current_frame_num / state['fps']
            
            # --- MODIFIED: START message with mm ---
            print(f"\n[START] Flame f-{flame_id} initiated at Frame {current_frame_num} ({initiation_time_s:.2f}s)")
            print(f"        Initial Position: {roi_center_x_mm:.3f} mm (Confidence: {best_match_score:.2f})\n")
            # --- END MODIFIED ---
            
            state['flame_data'][flame_id] = {
                # --- MODIFIED: Added Frame_ID to trajectory data ---
                'trajectory_points': [(current_frame_num, initiation_time_s, roi_center_x_mm)],
                'end_time_s': None
            }
            # --- MODIFIED: Using _mm variable ---
            state['leading_edge_pos_mm'] = roi_center_x_mm 
            state['tracker'] = cv2.TrackerCSRT_create()
            # --- MODIFIED: Must init tracker on the *original* (binarized) frame ---
            state['tracker'].init(frame, best_match_roi) 
            state['is_tracking'] = True
            state['lost_since_frame'] = 0
            state['last_known_box'] = [int(v) for v in best_match_roi]
            state['display_box'] = [int(v) for v in display_box]
            
def process_tracking_mode(frame, state, templates, f_dims, ignition_zone_pixels):
    """Updates the tracker, enforces forward-only movement, and handles tracking success or failure."""
    frame_width, frame_height = f_dims
    iz_x, iz_y, iz_w, iz_h = ignition_zone_pixels
    track_success, box = False, None
    is_redetection_frame = USE_PERIODIC_REDETECTION and (state['frame_counter'] % REDETECTION_INTERVAL == 0)

    # --- MODIFIED: Frame is binarized, use it for redetection ---
    if is_redetection_frame and state['last_known_box']:
        lx, ly, lw, lh = state['last_known_box']
        sw, sh = int(lw * SEARCH_AREA_SCALE), int(lh * SEARCH_AREA_SCALE)
        sx, sy = max(0, int(lx + lw/2 - sw/2)), max(0, int(ly + lh/2 - sh/2))
        new_box = re_detect_flame(frame, templates, (sx, sy, sw, sh))
        if new_box:
            state['tracker'] = cv2.TrackerCSRT_create()
            state['tracker'].init(frame, new_box)
            track_success, box = True, new_box
    if not track_success:
        track_success, box = state['tracker'].update(frame)

    current_frame_num = state['frame_counter'] # Get frame number for logging

    if track_success:
        x, y, w, h = [int(v) for v in box]
        is_in_frame = (x >= 0 and y >= 0 and x + w <= frame_width and y + h <= frame_height)
        is_in_ignition_zone = (x >= iz_x and y >= iz_y and x + w <= iz_x + iz_w and y + h <= iz_y + iz_h)
        if not (is_in_frame and is_in_ignition_zone):
            track_success = False
            # --- REMOVED: Per-frame terminal log ---

    if track_success:
        state['lost_since_frame'] = 0
        current_box = [int(v) for v in box]
        x, y, w, h = current_box
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            # --- MODIFIED: Frame is already B/W ---
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
            contours, _ = cv2.findContours(gray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_FLAME_AREA_PIXELS:
                    tx, ty, tw, th = cv2.boundingRect(largest_contour)
                    current_box = [x + tx, y + ty, tw, th]
        
        final_roi_box = current_box
        max_score = -1
        final_roi = frame[final_roi_box[1]:final_roi_box[1]+final_roi_box[3], final_roi_box[0]:final_roi_box[0]+final_roi_box[2]]
        if final_roi.size > 0:
            # --- MODIFIED: Frame is already B/W ---
            if len(final_roi.shape) == 3:
                final_roi_gray = cv2.cvtColor(final_roi, cv2.COLOR_BGR2GRAY)
            else:
                final_roi_gray = final_roi

            for t in templates:
                th, tw = t.shape
                if not (final_roi_gray.shape[0] < th or final_roi_gray.shape[1] < tw):
                    res = cv2.matchTemplate(final_roi_gray, t, cv2.TM_CCOEFF_NORMED)
                    _, s, _, _ = cv2.minMaxLoc(res)
                    if s > max_score: max_score = s
        
        if max_score >= EXTINGUISH_THRESHOLD:
            state['last_known_box'] = final_roi_box
            center_x_c1 = final_roi_box[0] + final_roi_box[2] / 2
            center_y_c1 = final_roi_box[1] + final_roi_box[3] / 2
            
            # --- MODIFIED: Using _mm variables ---
            c1_pos_mm = get_centroid_x_in_mm((center_x_c1, center_y_c1, 0, 0), PIXELS_PER_MM, state['direction'], state['total_width_mm'])
            
            # --- REMOVED: Per-frame terminal log ---
            
            if c1_pos_mm is not None:
                # --- MODIFIED: Using _mm variable ---
                if c1_pos_mm > state['leading_edge_pos_mm']:
                    state['leading_edge_pos_mm'] = c1_pos_mm
                    
                    db_w, db_h = DISPLAY_BOX_SIZE
                    dx = int(center_x_c1 - db_w / 2)
                    dy = int(center_y_c1 - db_h / 2)
                    state['display_box'] = (dx, dy, db_w, db_h)
        else:
            # --- REMOVED: Per-frame terminal log ---
            if state['lost_since_frame'] == 0: state['lost_since_frame'] = state['frame_counter']
    else:
        # --- REMOVED: Per-frame terminal log ---
        if state['lost_since_frame'] == 0: state['lost_since_frame'] = state['frame_counter']

    if state['is_tracking']:
        if state['display_box']:
            db_x, db_y, db_w, db_h = state['display_box']
            # --- MODIFIED: Color tuple for B/W image (use shades of gray) ---
            color = (100) if state['lost_since_frame'] > 0 else (255)
            text = "Lost..." if state['lost_since_frame'] > 0 else f"f-{state['flame_id_counter']} (Tracking)"
            
            # Draw on the 3-channel BGR frame `frame`
            cv2.rectangle(frame, (db_x, db_y), (db_x + db_w, db_y + db_h), color, 2)
            cv2.putText(frame, text, (db_x, db_y - 10), FONT, 0.6, color, 2)

        seconds_lost = 0
        if state['lost_since_frame'] > 0:
            seconds_lost = (state['frame_counter'] - state['lost_since_frame']) / state['fps']

        is_within_recording_window = (state['lost_since_frame'] == 0) or (seconds_lost <= RECORDING_CUTOFF_S)

        if is_within_recording_window:
            flame_id = state['flame_id_counter']
            # --- MODIFIED: Read new tuple structure ---
            last_frame, last_saved_time, last_saved_pos = state['flame_data'][flame_id]['trajectory_points'][-1]
            
            current_time_s = state['frame_counter'] / state['fps']
            current_pos_for_csv = state['leading_edge_pos_mm'] # --- MODIFIED: _mm variable
            distance_moved = abs(current_pos_for_csv - last_saved_pos)
            time_elapsed = current_time_s - last_saved_time
            
            # --- MODIFIED: Using _mm variable ---
            if (distance_moved >= MIN_RECORD_DISTANCE_MM) or (time_elapsed >= MAX_TIME_INTERVAL_S):
                # --- MODIFIED: Save Frame_ID, time, and pos ---
                state['flame_data'][flame_id]['trajectory_points'].append((current_frame_num, current_time_s, current_pos_for_csv))

        if seconds_lost > LOST_TIMEOUT_SECONDS:
            # --- MODIFIED: END message with mm ---
            flame_id = state['flame_id_counter']
            last_pos_mm = state['leading_edge_pos_mm']
            lost_at_frame = state['lost_since_frame']
            current_time_s = (lost_at_frame / state['fps']) + LOST_TIMEOUT_SECONDS

            print(f"\n[END] Flame f-{flame_id} extinguished (Timeout).")
            print(f"        Last seen at Frame {lost_at_frame} ({current_time_s:.2f}s)")
            print(f"        Final Position: {last_pos_mm:.3f} mm\n")
            # --- END MODIFIED ---
            
            state['flame_data'][flame_id]['end_time_s'] = current_time_s
            state['is_tracking'] = False

# --- MODIFIED: Updated function to save new CSV format ---
def save_flame_data(state):
    """Saves all collected flame trajectory data to a CSV file."""
    if not state['flame_data']: return
    
    # --- NEW: Ensure the output directory exists ---
    output_dir = os.path.dirname(TRAJECTORY_OUTPUT_FILE)
    if output_dir: # Check if it's not an empty string
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        with open(TRAJECTORY_OUTPUT_FILE, 'w', newline='') as f:
            # --- MODIFIED: New header ---
            f.write("Frame_ID,Flame_ID,Timestamp_s,Position_MM\n")
            for flame_id, data in sorted(state['flame_data'].items()):
                
                # --- MODIFIED: Unpack new tuple (frame_id, timestamp, position) ---
                for frame_id, timestamp, position in data['trajectory_points']:
                    # --- MODIFIED: Write new format ---
                    f.write(f"{frame_id},{flame_id},{timestamp:.6f},{position}\n")

        print(f"\nDetailed flame trajectory data saved to '{TRAJECTORY_OUTPUT_FILE}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save data: {e}")

# --- MODIFIED: Updated draw_scale_bar for mm ---
def draw_scale_bar(image, scale_info):
    """Draws a full-width, light grey scale bar on an image with directional numbering."""
    # --- MODIFIED: Use a bright color for B/W image ---
    SCALE_COLOR = (192) # Light Grey (will be visible on black)
    pixels_per_mm = scale_info['pixels_per_mm'] # --- MODIFIED: _mm
    y_pos = scale_info['scale_y_position']
    direction = scale_info['direction']
    
    # Check if image is 3-channel (BGR) or 1-channel (Grayscale)
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        SCALE_COLOR = 192 # Use single value for grayscale
        
    if height == 0 or width == 0: return
    
    cv2.line(image, (0, y_pos), (width - 1, y_pos), SCALE_COLOR, 1)
    
    num_ticks = int(width / pixels_per_mm) + 1 # --- MODIFIED: _mm
    
    for i in range(num_ticks):
        label_num = i
        
        current_x = -1
        if direction == 'L':
            current_x = int(i * pixels_per_mm) # --- MODIFIED: _mm
        else: # 'R'
            current_x = (width - 1) - int(i * pixels_per_mm) # --- MODIFIED: _mm
        
        if current_x < 0 or current_x >= width:
            continue 
            
        cv2.line(image, (current_x, y_pos - 2), (current_x, y_pos + 2), SCALE_COLOR, 1)
        
        # --- MODIFIED: Label every 50mm (was 5cm) ---
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
            
            # --- MODIFIED: Added 'mm' to label ---
            cv2.putText(image, f"{label_text} mm", (text_x_pos, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, SCALE_COLOR, 1)

def main():
    """Main function to run the flame tracking process."""
    
    # --- 1. GET USER INPUT FOR DIRECTION ---
    tracking_direction = ''
    while tracking_direction not in ['L', 'R']:
        tracking_direction = input("Enter tracking direction (L for LTR, R for RTL): ").upper()
        if tracking_direction not in ['L', 'R']:
            print("Invalid input. Please enter 'L' or 'R'.")
    
    print(f"Tracking direction set to: {'Left-to-Right' if tracking_direction == 'L' else 'Right-to-Left'}")
    
    # --- 2. INITIALIZE SYSTEM FROM IMAGES ---
    image_paths, templates, frame_width, frame_height, fps = initialize_system_from_images(
        IMAGE_FOLDER_PATH_INPUT, 
        TEMPLATE_FOLDER_PATH, 
        FPS_FOR_TIMESTAMPS
    )
    
    # --- 3. CONVERT ALL MM VALUES TO PIXELS ---
    # --- MODIFIED: All variables renamed to _mm ---
    # Ignition zone for *tracker*
    iz_x_start_px, iz_y_start_px = int(IGNITION_ZONE_MM[0] * PIXELS_PER_MM), int(IGNITION_ZONE_MM[1] * PIXELS_PER_MM)
    iz_x_end_px, iz_y_end_px = int(IGNITION_ZONE_MM[2] * PIXELS_PER_MM), int(IGNITION_ZONE_MM[3] * PIXELS_PER_MM)
    ignition_zone_pixels = (iz_x_start_px, iz_y_start_px, iz_x_end_px - iz_x_start_px, iz_y_end_px - iz_y_start_px)
    
    # Close zone for *background subtractor*
    close_x1 = int(CLOSE_ZONE_RIGHT_PIPE_MM[0] * PIXELS_PER_MM)
    close_y1 = int(CLOSE_ZONE_RIGHT_PIPE_MM[1] * PIXELS_PER_MM)
    close_x2 = int(CLOSE_ZONE_RIGHT_PIPE_MM[2] * PIXELS_PER_MM)
    close_y2 = int(CLOSE_ZONE_RIGHT_PIPE_MM[3] * PIXELS_PER_MM)
    close_region_pixels = (close_x1, close_y1, close_x2, close_y2)

    # Scale bar position
    scale_y_position_pixels = int(scale_y_position_mm * PIXELS_PER_MM)

    # Scale bar parameters dictionary
    scale_parameters = {
        "pixels_per_mm": PIXELS_PER_MM, # --- MODIFIED: _mm
        "scale_y_position": scale_y_position_pixels,
        "direction": tracking_direction,
    }

    # --- 4. SETUP VIDEO OUTPUT ---
    os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
    video_basename = os.path.basename(os.path.normpath(IMAGE_FOLDER_PATH_INPUT))
    # --- MODIFIED: Updated output video name ---
    VIDEO_OUTPUT_PATH = os.path.join(VIDEO_OUTPUT_FOLDER, f"{video_basename}_NEWflame_tracked.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') # Codec for .mp4

    # --- NEW: Set different FPS for output video (Video Clock) ---
    total_frames = len(image_paths)
    # Calculate video FPS for a 5-minute (300 second) duration
    OUTPUT_VIDEO_FPS = total_frames / 300.0  
    print(f"--- Output video will be saved at {OUTPUT_VIDEO_FPS:.2f} FPS (5 min duration) ---")

    # --- NEW: Calculate dynamic playback delay ---
    #DYNAMIC_PLAYBACK_DELAY = 1
    DYNAMIC_PLAYBACK_DELAY = int(1000 / OUTPUT_VIDEO_FPS)
    if DYNAMIC_PLAYBACK_DELAY < 1: 
        DYNAMIC_PLAYBACK_DELAY = 1 # cv2.waitKey() must be at least 1
    print(f"--- Preview window playback delay set to: {DYNAMIC_PLAYBACK_DELAY} ms ---")

    # --- MODIFIED: Use the new OUTPUT_VIDEO_FPS for the video file ---
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, OUTPUT_VIDEO_FPS, (frame_width, frame_height), isColor=True)
    
    # --- 5. MODIFIED: INITIALIZE STATIC BACKGROUND SUBTRACTION ---
    print(f"Applying MORPH_CLOSE to region: {CLOSE_ZONE_RIGHT_PIPE_MM}") # --- MODIFIED: _mm
    print("Applying a stronger MORPH_OPEN to all other regions.")
    print(f"Using STATIC background subtraction from frame index {FRAME_FOR_BACKGROUND}.")

    # --- Load the specific background frame ---
    if FRAME_FOR_BACKGROUND >= len(image_paths):
        print(f"[ERROR] FRAME_FOR_BACKGROUND index ({FRAME_FOR_BACKGROUND}) is out of range. Max index is {len(image_paths) - 1}.")
        sys.exit()
        
    bg_image_path = image_paths[FRAME_FOR_BACKGROUND]
    bg_frame = cv2.imread(bg_image_path)
    if bg_frame is None:
        print(f"[ERROR] Could not read background frame at: {bg_image_path}")
        sys.exit()
        
    # Convert background to grayscale and blur it
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
    
    # --- CHANGE 1: Create the static object mask ---
    # This mask will be white (255) for any pixel in the background
    # frame that has an intensity > 1.
    _, static_object_mask = cv2.threshold(bg_gray, 5, 255, cv2.THRESH_BINARY)
    # --- END CHANGE 1 ---
    
    print(f"--- Successfully loaded background from: {bg_image_path} ---")
    print("--- Created static object mask from background frame ---") # Added log

    # Create masks for dual morphology
    close_mask = np.zeros((frame_height, frame_width), dtype="uint8")
    cx1, cy1, cx2, cy2 = close_region_pixels
    cv2.rectangle(close_mask, (cx1, cy1), (cx2, cy2), 255, -1)
    open_mask = cv2.bitwise_not(close_mask)

    # --- REMOVED: MOG2 Subtractors ---

    # Morphological kernels (still needed for cleanup)
    close_kernel = np.ones((3, 3), np.uint8)
    close_iterations = 10
    open_kernel = np.ones((20, 20), np.uint8)
    open_iterations = 40

    # --- 6. INITIALIZE TRACKER STATE ---
    # --- MODIFIED: Renamed variables to _mm ---
    TOTAL_WIDTH_MM = (frame_width - 1) / PIXELS_PER_MM
    
    display_box_area = DISPLAY_BOX_SIZE[0] * DISPLAY_BOX_SIZE[1]
    min_flame_area_check = display_box_area / 800.0
    print(f"--- Min start area check: {min_flame_area_check}px (DISPLAY_BOX_SIZE_AREA / 5.0) ---")
    
    state = {
        'tracker': None, 'is_tracking': False, 'flame_id_counter': 0,
        'last_known_box': None, 'lost_since_frame': 0, 'frame_counter': 0,
        'flame_data': {}, 'fps': fps, # This is the DATA CLOCK (30.0)
        'display_box': None,
        'leading_edge_pos_mm': 0.0, # --- MODIFIED: _mm
        'direction': tracking_direction,
        'total_width_mm': TOTAL_WIDTH_MM, # --- MODIFIED: _mm
        'min_flame_area_check': min_flame_area_check
    }

    print("\nPress 'q' to quit.")
    print("---------------------------------------------------")
    print("--- STARTING FLAME TRACKING ---")
    
    # --- 7. MAIN PROCESSING LOOP ---
    for image_path in image_paths:
        # --- Check if this is the background frame; skip if it is ---
        if image_path == bg_image_path:
            # print(f"Skipping background frame: {os.path.basename(image_path)}") # This is a bit noisy
            continue
            
        frame_original = cv2.imread(image_path)
        if frame_original is None:
            print(f"[WARN] Could not read image {image_path}, skipping.")
            continue
            
        state['frame_counter'] += 1
        
        # --- START: MODIFIED BACKGROUND SUBTRACTION LOGIC ---
        
        # Convert current frame to grayscale and blur
        current_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

        # 1. Calculate Absolute Difference
        diff_mask = cv2.absdiff(bg_gray, current_gray)
        
        # 2. Threshold the difference
        ret, single_fg_mask = cv2.threshold(diff_mask, STATIC_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # --- CHANGE 2: REMOVED SUBTRACTION FROM HERE ---
        # The line 'single_fg_mask = cv2.subtract(single_fg_mask, static_object_mask)'
        # has been deleted from this position.
        # --- END CHANGE 2 ---
        
        # --- END: MOG2 logic removed ---

        # --- START: Re-used cleanup and binarization logic ---
        
        # Get the "flame" from the SENSITIVE mask
        fg_in_close_zone = cv2.bitwise_and(single_fg_mask, single_fg_mask, mask=close_mask)
        # Get the "background" from the NORMAL (less noisy) mask
        fg_in_open_zone = cv2.bitwise_and(single_fg_mask, single_fg_mask, mask=open_mask)

        processed_close_zone = cv2.morphologyEx(fg_in_close_zone, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
        processed_open_zone = cv2.morphologyEx(fg_in_open_zone, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)

        fg_mask_cleaned = cv2.bitwise_or(processed_close_zone, processed_open_zone)
        
        # --- CONTRAST ENHANCEMENT ---
        flame_region_orig_color = cv2.bitwise_and(frame_original, frame_original, mask=fg_mask_cleaned)
        hsv_flame = cv2.cvtColor(flame_region_orig_color, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_flame)
        v = cv2.convertScaleAbs(v, alpha=CONTRAST_FACTOR, beta=0) 
        merged_hsv = cv2.merge([h, s, v])
        enhanced_flame_bgr = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
        
        # --- BINARIZATION ---
        gray_enhanced_flame = cv2.cvtColor(enhanced_flame_bgr, cv2.COLOR_BGR2GRAY)
        ret, binary_mask = cv2.threshold(gray_enhanced_flame, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # --- CHANGE 3: Apply static mask at the VERY END ---
        # As you requested, we take the final binary_mask and
        # subtract the static_object_mask from it.
        # This "pokes holes" and sets all static object
        # positions to black (0).
        binary_mask = cv2.subtract(binary_mask, static_object_mask)
        # --- END CHANGE 3 ---
        
        # This is the final frame to be used for tracking and output
        # --- MODIFIED: Draw on BGR frame, but track on B/W ---
        # We create a 3-channel BGR version for drawing boxes, text, etc.
        frame_to_process = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        # --- END: BACKGROUND SUBTRACTION LOGIC ---

        
        # --- START: TRACKING LOGIC (using 'frame_to_process') ---
        iz_x, iz_y, iz_w, iz_h = ignition_zone_pixels
        # --- MODIFIED: Draw ignition zone in gray ---
        cv2.rectangle(frame_to_process, (iz_x, iz_y), (iz_x + iz_w, iz_y + iz_h), (200, 200, 200), 1)
        
        if state['is_tracking']:
            # We pass frame_to_process (BGR) for both tracking and drawing
            process_tracking_mode(frame_to_process, state, templates, (frame_width, frame_height), ignition_zone_pixels)
        else:
            process_detection_mode(frame_to_process, state, templates, ignition_zone_pixels)
        
        # --- MODIFIED: Using Data Clock (state['fps']) for time display ---
        current_time_s = state['frame_counter'] / state['fps'] 
        # --- MODIFIED: Draw text in white ---
        cv2.putText(frame_to_process, f"Time: {current_time_s:.2f}s", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        
        # --- ADD SCALE BAR ---
        draw_scale_bar(frame_to_process, scale_parameters)
        
        # --- WRITE AND SHOW FRAME ---
        out.write(frame_to_process)
        cv2.imshow("Automated Flame Tracker", frame_to_process)
        
        # --- MODIFIED: Use new dynamic delay ---
        if cv2.waitKey(DYNAMIC_PLAYBACK_DELAY) & 0xFF == ord('q'):
            print("--- User quit processing ---")
            break
            
    # --- 8. CLEANUP ---
    if state['is_tracking'] and state['flame_id_counter'] > 0 and state['flame_data'][state['flame_id_counter']]['end_time_s'] is None:
        flame_id = state['flame_id_counter']
        final_time_s = state['frame_counter'] / state['fps']
        last_pos_mm = state['leading_edge_pos_mm'] # --- MODIFIED: _mm
        
        # --- MODIFIED: END message with mm ---
        print(f"\n[END] Flame f-{flame_id} still tracking at end of video.")
        print(f"        Final Frame: {state['frame_counter']} ({final_time_s:.2f}s)")
        print(f"        Final Position: {last_pos_mm:.3f} mm\n")
        # --- END NEW ---
        
        state['flame_data'][flame_id]['end_time_s'] = final_time_s

    save_flame_data(state)
    
    out.release()
    cv2.destroyAllWindows()
    
    print("\nProcessing complete.")
    print(f"Tracking video saved to: {VIDEO_OUTPUT_PATH}")

if __name__ == "__main__":
    main()