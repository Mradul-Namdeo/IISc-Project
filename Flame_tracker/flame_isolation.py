import cv2
import numpy as np
import sys

def isolate_flame_in_video(input_video_path, output_video_path, pipe_coords):
    """
    Isolates a moving flame within a specified region of a video using MOG2
    background subtraction and saves the result to a new file.

    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {input_video_path}")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    roi_mask = np.zeros((frame_height, frame_width), dtype="uint8")
    x1, y1, x2, y2 = pipe_coords
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)

    subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

    print(f"Processing '{input_video_path}'...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        fg_mask = subtractor.apply(masked_frame)

        kernel = np.ones((3,3), np.uint8)
        fg_mask_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask_cleaned = cv2.dilate(fg_mask_cleaned, kernel, iterations=1)
        
        result = cv2.bitwise_and(frame, frame, mask=fg_mask_cleaned)
        
        out.write(result)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  ... processed {frame_count} frames.")

    print(f"\nProcessing complete. Output video saved to: {output_video_path}")
    cap.release()
    out.release()
    return True

# --- Example Usage ---
if __name__ == '__main__':
    input_file = r'/content/U_0p3_Phi_1p0.mp4'
    output_file = r'/content/U_0p3_Phi_1p0_background_subtracted.mp4'
    pipe_region = (400, 0, 800, 200)

    success = isolate_flame_in_video(input_file, output_file, pipe_region)

    if not success:
        print("\nAn error occurred during video processing.")
