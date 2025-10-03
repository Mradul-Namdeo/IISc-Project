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
TRAJECTORY_OUTPUT_FILE = "flame_trajectories.csv"

# --- Physical Setup (in cm) ---
IGNITION_ZONE_CM = (4.0, 0.2, 25, 2.3)
PIPE_ZONE_CM = (0.0, 0.1, 25.5, 2.4)
START_POSITION_CM = 4.5
START_GATE_WIDTH_CM = 1.0
PIXELS_PER_CM = 50

# --- Tracking Parameters ---
CONFIDENCE_THRESHOLD = 0.3
EXTINGUISH_THRESHOLD = -2
LOST_TIMEOUT_SECONDS = 12.0

# --- Advanced Refinement ---
USE_PERIODIC_REDETECTION = True
REDETECTION_INTERVAL = 15
SEARCH_AREA_SCALE = 1.5
FLAME_BRIGHTNESS_THRESHOLD = 100
MIN_FLAME_AREA_PIXELS = 10
MIN_RECORD_DISTANCE_CM = 0.2 # 2 millimeters

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


def initialize_system():
    """Loads video, templates, and calculates initial pixel coordinates."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
        sys.exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load templates
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
    return cap, templates, frame_width, frame_height


def process_detection_mode(frame, state, templates, iz_coords):
    """Detects a new flame within the ignition zone."""
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
        gate_min_cm = START_POSITION_CM - (START_GATE_WIDTH_CM / 2)
        gate_max_cm = START_POSITION_CM + (START_GATE_WIDTH_CM / 2)

        if gate_min_cm <= roi_center_x_cm <= gate_max_cm:
            state['flame_id_counter'] += 1
            flame_id = state['flame_id_counter']
            print(f"--- Flame f-{flame_id} initiated at x-centroid: {roi_center_x_cm} cm (Confidence: {best_match_score:.2f}) ---")

            state['flame_trajectories'][flame_id] = [roi_center_x_cm]
            state['tracker'] = cv2.TrackerCSRT_create()
            state['tracker'].init(frame, best_match_roi)
            state['is_tracking'] = True
            state['lost_timestamp'] = 0
            state['last_known_box'] = [int(v) for v in best_match_roi]
            state['frame_counter'] = 0


def process_tracking_mode(frame, state, templates, f_dims, iz_x_coords):
    """Tracks an existing flame, with refinement and re-detection."""
    frame_width, frame_height = f_dims
    iz_x_start, iz_x_end = iz_x_coords

    # --- Step 1: Update tracker position (with periodic re-detection) ---
    is_redetection_frame = USE_PERIODIC_REDETECTION and (state['frame_counter'] % REDETECTION_INTERVAL == 0)
    track_success, box = False, None

    if is_redetection_frame and state['last_known_box']:
        # Attempt to re-detect for higher accuracy
        lx, ly, lw, lh = state['last_known_box']
        sw, sh = int(lw * SEARCH_AREA_SCALE), int(lh * SEARCH_AREA_SCALE)
        sx, sy = max(0, int(lx + lw/2 - sw/2)), max(0, int(ly + lh/2 - sh/2))

        search_area = frame[sy:min(sy+sh, frame_height), sx:min(sx+sw, frame_width)]
        if search_area.size > 0:
            search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
            best_score, best_roi_rel = -1, None
            for t in templates:
                th, tw = t.shape
                if not (search_gray.shape[0] < th or search_gray.shape[1] < tw):
                    res = cv2.matchTemplate(search_gray, t, cv2.TM_CCOEFF_NORMED)
                    val, _, loc, _ = cv2.minMaxLoc(res)
                    if val > best_score:
                        best_score, best_roi_rel = val, (loc[0], loc[1], tw, th)

            if best_score >= CONFIDENCE_THRESHOLD:
                rx, ry, rw, rh = best_roi_rel
                abs_roi = (rx + sx, ry + sy, rw, rh)
                state['tracker'] = cv2.TrackerCSRT_create()
                state['tracker'].init(frame, abs_roi)
                track_success, box = True, abs_roi

    if not track_success: # Fallback to normal tracking if re-detection fails or is skipped
        track_success, box = state['tracker'].update(frame)

    # --- Step 2: Validate the tracked box ---
    if track_success:
        x, y, w, h = [int(v) for v in box]
        if not (x >= 0 and y >= 0 and x + w <= frame_width and y + h <= frame_height):
            track_success = False
        if x < iz_x_start or (x + w) > iz_x_end:
            track_success = False

    # --- Step 3: Process the result (success or failure) ---
    if track_success:
        state['lost_timestamp'] = 0
        current_box = [int(v) for v in box]

        # --- Refine box with contour analysis ---
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

        # --- Verify flame presence with template matching ---
        x,y,w,h = current_box
        final_roi = frame[y:y+h, x:x+w]
        max_score = -1
        if final_roi.size > 0:
            final_roi_gray = cv2.cvtColor(final_roi, cv2.COLOR_BGR2GRAY)
            for t in templates:
                th, tw = t.shape
                if not (final_roi_gray.shape[0] < th or final_roi_gray.shape[1] < tw):
                    res = cv2.matchTemplate(final_roi_gray, t, cv2.TM_CCOEFF_NORMED)
                    _, s, _, _ = cv2.minMaxLoc(res)
                    if s > max_score: max_score = s

        if max_score >= EXTINGUISH_THRESHOLD:
            state['last_known_box'] = current_box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"f-{state['flame_id_counter']} (Score: {max_score:.2f})", (x, y-10), FONT, 0.6, (0,255,0), 2)

            # Record trajectory
            pos_x = get_centroid_x_in_cm(current_box, PIXELS_PER_CM)
            fid = state['flame_id_counter']
            if pos_x is not None and abs(pos_x - state['flame_trajectories'][fid][-1]) >= MIN_RECORD_DISTANCE_CM:
                state['flame_trajectories'][fid].append(pos_x)
        else:
            pos_x = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
            print(f"Flame f-{state['flame_id_counter']} faded. Extinguished at x-centroid: {pos_x} cm")
            state['is_tracking'] = False
    else:
        # --- Handle tracking failure ---
        if state['lost_timestamp'] == 0: state['lost_timestamp'] = time.time()

        if (time.time() - state['lost_timestamp']) > LOST_TIMEOUT_SECONDS:
            pos_x = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
            print(f"Flame f-{state['flame_id_counter']} lost. Extinguished at x-centroid: {pos_x} cm")
            state['is_tracking'] = False
        elif state['last_known_box']:
            x,y,w,h = state['last_known_box']
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, f"f-{state['flame_id_counter']} Lost...", (x, y-10), FONT, 0.6, (0,0,255), 2)


def save_trajectories(state):
    """Saves the recorded flame trajectory data to a CSV file."""
    if not state['flame_trajectories']:
        return
    try:
        with open(TRAJECTORY_OUTPUT_FILE, 'w') as f:
            f.write("Flame_ID,X_Centroid_Positions_CM\n")
            for flame_id, positions in state['flame_trajectories'].items():
                positions_str = ",".join(map(str, positions))
                f.write(f"{flame_id},{positions_str}\n")
        print(f"\nFlame trajectories saved to '{TRAJECTORY_OUTPUT_FILE}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save trajectory data: {e}")


def main():
    """Main function to run the flame tracker."""
    cap, templates, frame_width, frame_height = initialize_system()

    # Calculate pixel coordinates from cm
    iz_x_start_px = int(IGNITION_ZONE_CM[0] * PIXELS_PER_CM)
    iz_y_start_px = int(IGNITION_ZONE_CM[1] * PIXELS_PER_CM)
    iz_x_end_px = int(IGNITION_ZONE_CM[2] * PIXELS_PER_CM)
    iz_y_end_px = int(IGNITION_ZONE_CM[3] * PIXELS_PER_CM)
    ignition_zone_pixels = (iz_x_start_px, iz_y_start_px, iz_x_end_px - iz_x_start_px, iz_y_end_px - iz_y_start_px)

    pipe_x1, pipe_y1 = int(PIPE_ZONE_CM[0] * PIXELS_PER_CM), int(PIPE_ZONE_CM[1] * PIXELS_PER_CM)
    pipe_x2, pipe_y2 = int(PIPE_ZONE_CM[2] * PIXELS_PER_CM), int(PIPE_ZONE_CM[3] * PIXELS_PER_CM)

    # Application state
    state = {
        'tracker': None,
        'is_tracking': False,
        'flame_id_counter': 0,
        'last_known_box': None,
        'lost_timestamp': 0,
        'frame_counter': 0,
        'flame_trajectories': {}
    }

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

        cv2.imshow("Automated Flame Tracker", frame)
        if cv2.waitKey(PLAYBACK_SPEED_DELAY) & 0xFF == ord('q'):
            break

    if state['is_tracking'] and state['last_known_box']:
        pos_x_cm = get_centroid_x_in_cm(state['last_known_box'], PIXELS_PER_CM)
        print(f"\nVideo ended. Final x-centroid of flame f-{state['flame_id_counter']}: {pos_x_cm} cm")

    save_trajectories(state)
    print("\nProcessing complete.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
