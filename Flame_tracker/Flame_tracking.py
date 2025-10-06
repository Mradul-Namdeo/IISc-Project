import cv2
import sys
import time
import glob
import os
import numpy as np

# ===================================================================================
# --- 1. CONFIGURATION ---
# ===================================================================================
# --- File Paths ---
VIDEO_PATH = r"D:\Flame_tracking\Flame_mp4_files\U_0p3_Phi_1p0_background_subtracted.mp4"
TEMPLATE_FOLDER_PATH = r"D:\Flame_tracking\Flame_similar_images"
TRAJECTORY_OUTPUT_FILE = "flame_analysis_data.csv"

# --- Physical Setup (in cm) ---
IGNITION_ZONE_CM = (4.0, 0.2, 25.5, 2.3)
PIPE_ZONE_CM = (0.0, 0.1, 25.5, 2.4)
PIXELS_PER_CM = 50

# Define the special "Holding Zone". If the flame is lost here,
# the tracker will wait indefinitely instead of timing out.
HOLDING_ZONE_CM = (4.5, 8.0)

ALLOWED_START_POSITIONS_CM = [5.0, 7.0, 8.0, 9.0, 10.0] 
START_POSITION_TOLERANCE_CM = 1.0

# --- Tracking Parameters ---
CONFIDENCE_THRESHOLD = 0.5
EXTINGUISH_THRESHOLD = -1.5
LOST_TIMEOUT_SECONDS = 3.0

# --- Advanced Refinement ---
USE_PERIODIC_REDETECTION = True
REDETECTION_INTERVAL = 15
SEARCH_AREA_SCALE = 1.5
FLAME_BRIGHTNESS_THRESHOLD = 100
MIN_FLAME_AREA_PIXELS = 10 
MIN_RECORD_DISTANCE_CM = 0.4

# --- Display Settings ---
PLAYBACK_SPEED_DELAY = 80
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ===================================================================================


def get_centroid_x_in_cm(box, pixels_per_cm):
    """Calculates the x-coordinate of a bounding box's center in centimeters."""
    if box is None:
        return None
    (x, y, w, h) = box
    center_x_px = x + w / 2
    return round(center_x_px / pixels_per_cm, 2)

def re_detect_flame(frame, templates, search_area_box):
    """
    Actively searches for a flame within a given search area using template matching.
    Returns the absolute bounding box of the best match if found, otherwise None.
    """
    sx, sy, sw, sh = search_area_box
    frame_height, frame_width, _ = frame.shape
    
    # Ensure the search area is within the frame boundaries
    search_area = frame[sy:min(sy+sh, frame_height), sx:min(sx+sw, frame_width)]
    
    if search_area.size == 0:
        return None

    search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
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
        # Return the box with absolute frame coordinates
        return (rx + sx, ry + sy, rw, rh)
        
    return None


def initialize_system():
    """Loads video, templates, and calculates initial pixel coordinates."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
        sys.exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[WARNING] Video FPS is 0. Defaulting to 30 FPS.")
        fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    template_paths = []
    for ext in image_extensions:
        search_path = os.path.join(TEMPLATE_FOLDER_PATH, ext)
        template_paths.extend(glob.glob(search_path))
    templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in template_paths]
    templates = [t for t in templates if t is not None]
    if not templates:
        print(f"[ERROR] No valid template images found in '{TEMPLATE_FOLDER_PATH}'. Exiting.")
        sys.exit()
    print(f"--- Flame Tracker: Loaded {len(templates)} templates from folder ---")
    print(f"--- Video Properties: {frame_width}x{frame_height} @ {fps:.2f} FPS ---")
    return cap, templates, frame_width, frame_height, fps


def process_detection_mode(frame, state, templates, iz_coords):
    """Detects a new flame within the ignition zone at one of the allowed positions."""
    iz_x, iz_y, iz_w, iz_h = iz_coords
    ignition_area = frame[iz_y:iz_y + iz_h, iz_x:iz_x + iz_w]
    ignition_area_gray = cv2.cvtColor(ignition_area, cv2.COLOR_BGR2GRAY)
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
        roi_center_x_cm = get_centroid_x_in_cm(best_match_roi, PIXELS_PER_CM)
        is_valid_start = False
        for start_pos_cm in ALLOWED_START_POSITIONS_CM:
            min_gate = start_pos_cm - (START_POSITION_TOLERANCE_CM / 2)
            max_gate = start_pos_cm + (START_POSITION_TOLERANCE_CM / 2)
            if min_gate <= roi_center_x_cm <= max_gate:
                is_valid_start = True
                break
        if is_valid_start:
            state['flame_id_counter'] += 1
            flame_id = state['flame_id_counter']
            initiation_time_s = state['frame_counter'] / state['fps']
            print(f"--- Flame f-{flame_id} initiated at {initiation_time_s:.2f}s (x: {roi_center_x_cm} cm, Conf: {best_match_score:.2f}) ---")
            state['flame_data'][flame_id] = {'trajectory': [roi_center_x_cm], 'start_time_s': initiation_time_s, 'end_time_s': None }
            state['tracker'] = cv2.TrackerCSRT_create()
            state['tracker'].init(frame, best_match_roi)
            state['is_tracking'] = True
            state['lost_since_frame'] = 0 
            state['last_known_box'] = [int(v) for v in best_match_roi]

def process_tracking_mode(frame, state, templates, f_dims, iz_x_coords):
    """Tracks an existing flame, with special handling for the holding zone."""
    frame_width, frame_height = f_dims
    iz_x_start, iz_x_end = iz_x_coords
    
    is_redetection_frame = USE_PERIODIC_REDETECTION and (state['frame_counter'] % REDETECTION_INTERVAL == 0)
    track_success, box = False, None
    
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

    if track_success:
        x, y, w, h = [int(v) for v in box]
        if not (x >= 0 and y >= 0 and x + w <= frame_width and y + h <= frame_height) or \
           x < iz_x_start or (x + w) > iz_x_end:
            track_success = False

    if track_success:
        state['lost_since_frame'] = 0
        current_box = [int(v) for v in box]
        
        x, y, w, h = current_box
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_roi, FLAME_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_FLAME_AREA_PIXELS:
                    tx, ty, tw, th = cv2.boundingRect(largest_contour)
                    current_box = [x + tx, y + ty, tw, th]
        x,y,w,h = current_box
        final_roi = frame[y:y+h, x:x+w]
        max_score = -1
        if final_roi.size > 0:
            final_roi_gray = cv2.cvtColor(final_roi, cv2.COLOR_BGR2GRAY)
            for t in templates:
                th, tw = t.shape
                if not (final_roi_gray.shape[0] < th or final_roi_gray.shape[1] < tw):
                    res = cv2.matchTemplate(final_roi_gray, t, cv2.TM_CCOEFF_NORMED)
                    # Correctly capture max_loc to prevent 'not defined' error
                    _, s, _, max_loc = cv2.minMaxLoc(res)
                    if s > max_score: max_score = s
        
        if max_score >= EXTINGUISH_THRESHOLD:
            state['last_known_box'] = current_box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"f-{state['flame_id_counter']} (Score: {max_score:.2f})", (x, y-10), FONT, 0.6, (0,255,0), 2)
            
            pos_x = get_centroid_x_in_cm(current_box, PIXELS_PER_CM)
            fid = state['flame_id_counter']

            # Check if flame reached the end of the pipe
            pipe_end_cm = PIPE_ZONE_CM[2]
            if pos_x is not None and pos_x >= pipe_end_cm:
                extinguish_time_s = state['frame_counter'] / state['fps']
                state['flame_data'][fid]['end_time_s'] = extinguish_time_s
                
                if abs(pos_x - state['flame_data'][fid]['trajectory'][-1]) >= MIN_RECORD_DISTANCE_CM:
                    state['flame_data'][fid]['trajectory'].append(pos_x)
                    
                print(f"Flame f-{fid} reached end of pipe. Extinguished at {extinguish_time_s:.2f}s (x: {pos_x} cm)")
                state['is_tracking'] = False
                return

            # Record trajectory if flame has not reached the end
            if pos_x is not None and abs(pos_x - state['flame_data'][fid]['trajectory'][-1]) >= MIN_RECORD_DISTANCE_CM:
                state['flame_data'][fid]['trajectory'].append(pos_x)
        else:
            # Flame faded
            extinguish_time_s = state['frame_counter'] / state['fps']
            fid = state['flame_id_counter']
            state['flame_data'][fid]['end_time_s'] = extinguish_time_s
            pos_x = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
            print(f"Flame f-{fid} faded. Extinguished at {extinguish_time_s:.2f}s (x: {pos_x} cm)")
            state['is_tracking'] = False
    else:
        # Tracking failure logic
        last_pos_cm = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
        
        # Check if inside holding zone
        if last_pos_cm is not None and HOLDING_ZONE_CM[0] <= last_pos_cm <= HOLDING_ZONE_CM[1]:
            lx, ly, lw, lh = state['last_known_box']
            search_box = (max(0, lx-lw//2), max(0, ly-lh//2), lw*2, lh*2)
            
            reacquired_box = re_detect_flame(frame, templates, search_box)
            if reacquired_box:
                state['tracker'] = cv2.TrackerCSRT_create()
                state['tracker'].init(frame, reacquired_box)
                state['last_known_box'] = [int(v) for v in reacquired_box]
                x,y,w,h = state['last_known_box']
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"f-{state['flame_id_counter']} Re-acquired", (x, y-10), FONT, 0.6, (0,255,0), 2)
                pos_x = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
                state['flame_data'][state['flame_id_counter']]['trajectory'].append(pos_x)
            else:
                x,y,w,h = state['last_known_box']
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 2)
                cv2.putText(frame, f"f-{state['flame_id_counter']} Holding...", (x, y-10), FONT, 0.6, (0, 255, 255), 2)
        else:
            # Normal lost behavior
            if state['lost_since_frame'] == 0:
                state['lost_since_frame'] = state['frame_counter']
            
            frames_lost = state['frame_counter'] - state['lost_since_frame']
            seconds_lost = frames_lost / state['fps']

            if seconds_lost > LOST_TIMEOUT_SECONDS:
                extinguish_time_s = state['frame_counter'] / state['fps']
                fid = state['flame_id_counter']
                state['flame_data'][fid]['end_time_s'] = extinguish_time_s
                pos_x = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
                print(f"Flame f-{fid} lost. Extinguished at {extinguish_time_s:.2f}s (x: {pos_x} cm)")
                state['is_tracking'] = False
            elif state['last_known_box']:
                x,y,w,h = state['last_known_box']
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                cv2.putText(frame, f"f-{state['flame_id_counter']} Lost for {seconds_lost:.1f}s...", (x, y-10), FONT, 0.6, (0,0,255), 2)


def save_flame_data(state):
    """Saves the recorded flame analysis data to a CSV file."""
    if not state['flame_data']:
        return
    try:
        with open(TRAJECTORY_OUTPUT_FILE, 'w') as f:
            f.write("Flame_ID,Start_Time_s,End_Time_s,X_Centroid_Positions_CM\n")
            for flame_id, data in state['flame_data'].items():
                start_time = f"{data['start_time_s']:.2f}"
                end_time = f"{data['end_time_s']:.2f}" if data['end_time_s'] is not None else "N/A"
                positions_str = ",".join(map(str, data['trajectory']))
                f.write(f"{flame_id},{start_time},{end_time},{positions_str}\n")
        print(f"\nFlame analysis data saved to '{TRAJECTORY_OUTPUT_FILE}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save data: {e}")


def main():
    """Main function to run the flame tracker."""
    cap, templates, frame_width, frame_height, fps = initialize_system()
    iz_x_start_px = int(IGNITION_ZONE_CM[0] * PIXELS_PER_CM)
    iz_y_start_px = int(IGNITION_ZONE_CM[1] * PIXELS_PER_CM)
    iz_x_end_px = int(IGNITION_ZONE_CM[2] * PIXELS_PER_CM)
    iz_y_end_px = int(IGNITION_ZONE_CM[3] * PIXELS_PER_CM)
    ignition_zone_pixels = (iz_x_start_px, iz_y_start_px, iz_x_end_px - iz_x_start_px, iz_y_end_px - iz_y_start_px)
    pipe_x1, pipe_y1 = int(PIPE_ZONE_CM[0] * PIXELS_PER_CM), int(PIPE_ZONE_CM[1] * PIXELS_PER_CM)
    pipe_x2, pipe_y2 = int(PIPE_ZONE_CM[2] * PIXELS_PER_CM), int(PIPE_ZONE_CM[3] * PIXELS_PER_CM)
    state = {'tracker': None, 'is_tracking': False, 'flame_id_counter': 0, 'last_known_box': None, 'lost_since_frame': 0, 'frame_counter': 0, 'flame_data': {}, 'fps': fps}
    print("Press 'q' to quit.")
    print("---------------------------------------------------\n")
    while True:
        success, frame = cap.read()
        if not success:
            break
        state['frame_counter'] += 1
        cv2.rectangle(frame, (pipe_x1, pipe_y1), (pipe_x2, pipe_y2), (255, 255, 255), 2)
        if state['is_tracking']:
            process_tracking_mode(frame, state, templates, (frame_width, frame_height), (iz_x_start_px, iz_x_end_px))
        else:
            process_detection_mode(frame, state, templates, ignition_zone_pixels)
        current_time_s = state['frame_counter'] / state['fps']
        cv2.putText(frame, f"Time: {current_time_s:.2f}s", (10, 30), FONT, 0.7, (255,255,0), 2)
        cv2.imshow("Automated Flame Tracker", frame)
        if cv2.waitKey(PLAYBACK_SPEED_DELAY) & 0xFF == ord('q'):
            break
    if state['is_tracking']:
        final_time_s = state['frame_counter'] / state['fps']
        fid = state['flame_id_counter']
        if state['flame_data'].get(fid) and state['flame_data'][fid]['end_time_s'] is None:
            state['flame_data'][fid]['end_time_s'] = final_time_s
            pos_x_cm = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
            print(f"\nVideo ended. Final x-centroid of flame f-{fid}: {pos_x_cm} cm at {final_time_s:.2f}s")
    save_flame_data(state)
    print("\nProcessing complete.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
