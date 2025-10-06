import cv2
import os
import sys

# ===================================================================================
# --- 1. CONFIGURATION ---
# ===================================================================================

# --- Create a list of video paths to process in order ---
video_paths = [
    r"D:\Flame_tracking\Flame_mp4_files\U_0p3_Phi_1p0_background_subtracted.mp4",
    r"D:\Flame_tracking\Flame_mp4_files\U_0p475_Phi_1p0_background_subtracted.mp4"
    # You can add more video paths here, e.g., video_path3, video_path4, etc.
]

# The FULL path to the folder where ALL cropped images will be saved
output_folder_path = r"D:\Flame_tracking\test"

# Define the target size for all saved templates
TARGET_TEMPLATE_SIZE = (53, 46) # (width, height) in pixels
# ===================================================================================


# --- 2. SETUP ---
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    print(f"Created folder: '{output_folder_path}'")

# --- MODIFICATION: Move image_counter outside the loop so it doesn't reset ---
image_counter = 1

print("\n--- Template Cropping Tool (with Auto-Resize) ---")
print(f"All saved templates will be resized to {TARGET_TEMPLATE_SIZE[0]}x{TARGET_TEMPLATE_SIZE[1]} pixels.")
print("   c        : Capture current frame and enter Crop Mode")
print("   SPACEBAR : Manually Play / Pause")
print("   q        : Quit current video (or application if no video is playing)")
print("--------------------------------------------------\n")


# --- 3. MAIN LOOP ---
# --- MODIFICATION: Loop through each video path in the list ---
for current_video_path in video_paths:
    
    print(f"--- NOW PROCESSING: {os.path.basename(current_video_path)} ---")
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {current_video_path}")
        print("Skipping to the next video...")
        continue # Skip to the next iteration of the for loop

    is_paused = False
    
    # This inner loop processes a single video
    while True:
        if not is_paused:
            success, frame = cap.read()
            if not success:
                print(f"End of video '{os.path.basename(current_video_path)}'.")
                break # Exit the inner while loop to move to the next video
        
        # This check is for the case where the video ends and we are waiting for 'q'
        if 'frame' not in locals() or not success:
             if cv2.waitKey(0) & 0xFF == ord('q'):
                 break
             continue

        display_frame = frame.copy()
        status_text = "PAUSED" if is_paused else "PLAYING"
        status_color = (0, 0, 255) if is_paused else (0, 255, 0)
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        window_title = f"Frame Cropping Tool - {os.path.basename(current_video_path)}"
        cv2.imshow(window_title, display_frame)
        key = cv2.waitKey(60) & 0xFF

        # --- Handle User Input ---
        if key == ord('q'):
            break # Exit the inner while loop to move to the next video
        elif key == ord(' '):
            is_paused = not is_paused
        elif key == ord('c'):
            is_paused = True
            print("\n--- Paused. Entered Continuous Cropping Mode ---")
            
            while True:
                roi = cv2.selectROI("Crop regions (Press ESC to exit)", frame, fromCenter=False, showCrosshair=True)
                
                if roi[2] > 0 and roi[3] > 0:
                    x, y, w, h = roi
                    cropped_image = frame[y:y+h, x:x+w]
                    resized_crop = cv2.resize(cropped_image, TARGET_TEMPLATE_SIZE)

                    file_name = f"Similar image {image_counter}.png"
                    file_path = os.path.join(output_folder_path, file_name)
                    
                    cv2.imwrite(file_path, resized_crop)
                    print(f"Saved resized template as '{file_path}'")
                    image_counter += 1 # Increment the shared counter
                else:
                    break # User pressed ESC
            
            cv2.destroyWindow("Crop regions (Press ESC to exit)")
            is_paused = False
            print("--- Exited Cropping Mode. Resuming video... ---\n")

    # --- MODIFICATION: Clean up for the video that just finished ---
    cap.release()
    cv2.destroyAllWindows()


# --- 4. CLEAN UP ---
print("\nAll videos have been processed. Exiting application.")
