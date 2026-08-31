import cv2
import numpy as np
import os
from tqdm import tqdm
import tifffile as tiff

# ==========================================
# 0. CONFIGURATION
# ==========================================
BASE_ACTUAL_DIR = r"D:\Image Segmentation Task\DATA\Actual"
BASE_BG_DIR = r"D:\Image Segmentation Task\DATA\BG" 
BASE_OUTPUT_DIR = r"D:\Image Segmentation Task\Final_PP4000_4\Batch_Norm_Results"

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================
def load_16bit_raw(path):
    try:
        img = tiff.imread(path)
        return img.astype(np.uint16)
    except Exception as e:
        print(f"\n[IMAGE LOAD ERROR] Could not read file: {path}")
        return None

def min_max_invert_16bit(img_16bit):
    """
    Applies EXACT local ((I_max_local - I_t) / (I_max_local - I_min_local)) * 65535 inversion.
    Processed frame-by-frame to avoid 16-bit global drift.
    """
    img_float = img_16bit.astype(np.float32)
    i_min_local = np.min(img_float)
    i_max_local = np.max(img_float)
    
    denominator = (i_max_local - i_min_local) + 1e-6
    normalized_float = ((i_max_local - img_float) / denominator) * 65535.0
    
    return normalized_float.astype(np.uint16)

def compute_average_normalized_bg(bg_folder):
    """Normalizes every BG frame locally, then computes the 16-bit average."""
    if not os.path.exists(bg_folder):
        print(f"  [WARNING] BG folder not found: {bg_folder}")
        return None

    bg_files = sorted([f for f in os.listdir(bg_folder) if f.lower().endswith('.tif')])
    if not bg_files:
        return None
    
    print(f"  Calculating Average Normalized BG from {len(bg_files)} frames...")
    
    # Initialize a float64 accumulator to prevent 16-bit overflow during addition
    first_bg = load_16bit_raw(os.path.join(bg_folder, bg_files[0]))
    accumulator = np.zeros(first_bg.shape, dtype=np.float64)
    
    for f in tqdm(bg_files, desc="  Processing BG", leave=False, dynamic_ncols=True):
        img = load_16bit_raw(os.path.join(bg_folder, f))
        if img is not None:
            norm_bg = min_max_invert_16bit(img)
            accumulator += norm_bg.astype(np.float64)
            
    # Divide by total frames and safely cast back to 16-bit integer
    average_bg = (accumulator / len(bg_files)).astype(np.uint16)
    return average_bg

# ==========================================
# 2. PIPELINE PER FOLDER
# ==========================================
def run_pipeline(input_folder, bg_folder, output_base):
    preprocessed_folder = os.path.join(output_base, "Preprocessed_Inverted")
    os.makedirs(preprocessed_folder, exist_ok=True)

    target_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.tif')])[:3000]
    if not target_files:
        print(f"  [ERROR] No target TIFF files found in {input_folder}")
        return

    # =======================================================
    # COMPUTE AVERAGE BG
    # =======================================================
    average_normalized_bg = compute_average_normalized_bg(bg_folder)
    if average_normalized_bg is None:
        print("  [WARNING] Skipping BG subtraction (No BG data found).")

    # =======================================================
    # THE FRAME LOOP (ONLY PREPROCESSING & INVERSION)
    # =======================================================
    for f in tqdm(target_files, desc="  Processing Target Frames", leave=False, dynamic_ncols=True):
        img = load_16bit_raw(os.path.join(input_folder, f))
        if img is None: continue
        
        # 1. EXACT Local Normalization Frame-by-Frame
        raw_inv = min_max_invert_16bit(img)

        # 2. Save directly to Preprocessed_Inverted
        if average_normalized_bg is not None:
            bg_subtracted = cv2.absdiff(average_normalized_bg, raw_inv)
            tiff.imwrite(os.path.join(preprocessed_folder, f), bg_subtracted)
        else:
            tiff.imwrite(os.path.join(preprocessed_folder, f), raw_inv)

    print(f"  Completed. Results saved in: {preprocessed_folder}")

# ==========================================
# 3. BATCH PROCESSING
# ==========================================
def batch_process():
    if not os.path.exists(BASE_ACTUAL_DIR):
        print(f"\n[FATAL ERROR] Check base folder path: \nActual: {BASE_ACTUAL_DIR}")
        return

    actual_folders = [f for f in os.listdir(BASE_ACTUAL_DIR) if os.path.isdir(os.path.join(BASE_ACTUAL_DIR, f))]

    for actual_name in actual_folders:
        input_folder = os.path.join(BASE_ACTUAL_DIR, actual_name)
        
        # Dynamically map to the corresponding BG folder by searching for the prefix
        prefix = actual_name.rsplit('_', 1)[0] 
        
        bg_folder = None
        if os.path.exists(BASE_BG_DIR):
            for candidate in os.listdir(BASE_BG_DIR):
                if candidate.startswith(prefix) and os.path.isdir(os.path.join(BASE_BG_DIR, candidate)):
                    bg_folder = os.path.join(BASE_BG_DIR, candidate)
                    break
        
        # Fallback just in case a match wasn't found
        if bg_folder is None:
            bg_folder = os.path.join(BASE_BG_DIR, actual_name)
        
        output_base = os.path.join(BASE_OUTPUT_DIR, f"{actual_name}_Output")

        print(f"\n" + "="*60)
        print(f"Processing Dataset: {actual_name}")
        print(f"="*60)
        
        run_pipeline(input_folder, bg_folder, output_base)

if __name__ == "__main__":
    batch_process()
