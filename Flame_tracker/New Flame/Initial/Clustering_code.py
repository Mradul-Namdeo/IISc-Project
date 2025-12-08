import cv2
import numpy as np
import sys
import os

# --- USER CONFIGURATION ---
'''
INPUT_VIDEO_PATH = r'D:\CNN based video tracking\Flame_mp4_files\Phi_1p0_u_0p3_C001H001S0001.mp4'
OUTPUT_VIDEO_PATH = r'D:\CNN based video tracking\Recorded tracking videos\Phi_1p0_u_0p3_C001H001S0001_recorded_(K=2 mean Clustering).mp4'
'''
INPUT_VIDEO_PATH = r'D:\CNN based video tracking\WIN_20251204_17_02_13_Pro.mp4'
OUTPUT_VIDEO_PATH = r'D:\CNN based video tracking\WIN_20251204_17_02_13_Pro_recorded (K=4 mean Clustering).mp4'

# CLUSTERING SETTINGS
K_CLUSTERS = 4 

# --- NEW TOGGLE ---
# True  = Track Changes (Movement/Flicker)
# False = Track Brightness (Raw Intensity)
ENABLE_FRAME_DIFFERENCING = True  
# --------------------------

# 1. LOAD VIDEO
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
ret, old_frame = cap.read()

if not ret:
    print(f"Error: Could not read video file at: {INPUT_VIDEO_PATH}")
    sys.exit()

# 2. DYNAMIC ROI SELECTION
print("Step 1: Select ROI on the window and press SPACE or ENTER.")
roi_tuple = cv2.selectROI("Select Flame ROI", old_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Flame ROI")

if roi_tuple == (0, 0, 0, 0):
    print("No ROI selected. Exiting.")
    sys.exit()

x_roi, y_roi, w_roi, h_roi = roi_tuple
print(f"Selected ROI: {roi_tuple}")

# Pre-processing
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
prev_roi_gray = old_gray[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

# Setup Video Writer
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (vid_w, vid_h * 2))

print(f"Tracking Started. Frame Differencing: {ENABLE_FRAME_DIFFERENCING}")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Safety check
    if y_roi+h_roi > frame_gray.shape[0] or x_roi+w_roi > frame_gray.shape[1]:
        break

    curr_roi_gray = frame_gray[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    # --- 1. LOGIC SWITCH ---
    if ENABLE_FRAME_DIFFERENCING:
        # Calculate Magnitude of Change
        process_img = cv2.absdiff(prev_roi_gray, curr_roi_gray)
        view_label = "View 1: Frame Difference (Movement)"
    else:
        # Use Raw Intensity (Bright vs Dark)
        process_img = curr_roi_gray
        view_label = "View 1: Raw Intensity (Brightness)"

    # Gaussian Blur to smooth noise (useful for both modes)
    process_blur = cv2.GaussianBlur(process_img, (3, 3), 0)

    # 2. DATA PREPARATION
    pixel_values = process_blur.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # 3. K-MEANS CLUSTERING
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    if len(pixel_values) >= K_CLUSTERS:
        _, labels, centers = cv2.kmeans(pixel_values, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 4. IDENTIFY FLAME GROUP (Highest Intensity or Highest Change)
        centers = np.uint8(centers)
        flame_group_index = np.argmax(centers) 
        max_val = centers[flame_group_index][0]

        # Threshold check
        # If Differencing: Check if movement > 30
        # If Raw: Check if brightness > 30 (adjust if flame is dim)
        if max_val > 5: 
            labels = labels.flatten()
            group_mask = (labels == flame_group_index).astype(np.uint8) * 255
            group_mask = group_mask.reshape((h_roi, w_roi))

            # 5. FIND BOUNDING BOX
            contours, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 5:
                    rx, ry, rw, rh = cv2.boundingRect(largest_contour)
                    gx, gy = rx + x_roi, ry + y_roi
                    
                    # Draw Box
                    cv2.rectangle(frame, (gx, gy), (gx + rw, gy + rh), (0, 0, 255), 2)
                    cv2.putText(frame, f"Val: {max_val}", (gx, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # --- VISUALIZATION STACK ---
    process_view = cv2.cvtColor(process_img, cv2.COLOR_GRAY2BGR)
    
    top_canvas = np.zeros_like(frame)
    top_canvas[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = process_view
    
    cv2.putText(top_canvas, view_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "View 2: K-Means Tracking", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    combined_view = np.vstack((top_canvas, frame))

    prev_roi_gray = curr_roi_gray.copy()
    out.write(combined_view)
    
    display_view = cv2.resize(combined_view, (0, 0), fx=0.8, fy=0.8)
    cv2.imshow('K-Means Tracking', display_view)
    
    if cv2.waitKey(30) == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
