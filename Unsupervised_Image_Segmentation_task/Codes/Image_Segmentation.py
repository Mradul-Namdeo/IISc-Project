import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile as tiff
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import scipy.signal

# ==========================================
# 0. CONFIGURATION
# ==========================================
BASE_ACTUAL_DIR = r"D:\Image Segmentation Task\DATA\Actual"
BASE_BG_DIR = r"D:\Image Segmentation Task\DATA\BG" 
BASE_OUTPUT_DIR = r"D:\Image Segmentation Task\Final_PP4000_4\Batch_Preprocessed_Results"

FRAGMENT_OVERLAP_THRESHOLD = 0.1
SHIFT_VALUE = 0.0  # Adjust this value to shift the threshold if needed

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

def normalize_to_8bit(img_16bit):
    img_float = img_16bit.astype(np.float32)
    i_min, i_max = img_float.min(), img_float.max()
    if i_max > i_min:
        return (((img_float - i_min) / (i_max - i_min + 1e-6)) * 255.0).astype(np.uint8)
    return np.zeros_like(img_16bit, dtype=np.uint8)

def apply_pro_panel_for_video(img_8bit, label):
    if len(img_8bit.shape) == 2:
        img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_8bit.copy()
        
    img_bgr = cv2.copyMakeBorder(img_bgr, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 20), (20, 20, 20), -1)
    cv2.putText(img_bgr, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    return img_bgr

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
# 2. THE DYNAMIC PIPELINE ENGINE
# ==========================================
def isolate_target_global(diff_img_16bit, prev_main_core, persistent_fragment_mask, local_jet_threshold, trained_gmm, sorted_indices):
    h, w = diff_img_16bit.shape
    pixel_values = diff_img_16bit.flatten().astype(np.float32)
    
    # ---------------------------------------------------------
    # PHASE 1: GMM VISUALIZATION MAPPING 
    # ---------------------------------------------------------
    probs = trained_gmm.predict_proba(pixel_values.reshape(-1, 1))
    sorted_probs = probs[:, sorted_indices]
    
    dominant_cluster = np.argmax(sorted_probs, axis=1)
    dom_reshaped = dominant_cluster.reshape((h, w))
    pixel_vals_2d = pixel_values.reshape((h, w))
    
    palettes = {
        0: ((20, 20, 150), (100, 100, 255)),   # C0: Red 
        1: ((150, 20, 150), (255, 100, 255)),  # C1: Purple 
        2: ((20, 150, 150), (100, 255, 255)),  # C2: Yellow 
        3: ((20, 150, 20), (100, 255, 100))    # C3: Green 
    }
    
    cluster_color_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id in range(4):
        mask = (dom_reshaped == c_id)
        if not np.any(mask): continue
        c_pixels = pixel_vals_2d[mask]
        c_min, c_max = c_pixels.min(), c_pixels.max()
        t = (c_pixels - c_min) / (c_max - c_min + 1e-6) 
        
        dark = np.array(palettes[c_id][0], dtype=np.float32)
        light = np.array(palettes[c_id][1], dtype=np.float32)
        mapped = dark + (light - dark) * t[:, np.newaxis]
        cluster_color_bgr[mask] = mapped.astype(np.uint8)

    # ---------------------------------------------------------
    # EXACT STEPS FROM PREVIOUS CODE (STEPS 4 to 12)
    # ---------------------------------------------------------
    
    # Step 4: Binarization
    binary_all = np.zeros((h, w), dtype=np.uint8)
    binary_all[diff_img_16bit > local_jet_threshold] = 255
    
    # Step 5: CCL Priority & Noise Map
    num_labels, ccl_labels, stats, _ = cv2.connectedComponentsWithStats(binary_all, connectivity=8)
    largest_filtered = np.zeros_like(binary_all)
    noise_map = np.zeros_like(binary_all)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        # Sort indices to check from largest area to smallest
        sorted_indices = np.argsort(areas)[::-1] + 1 
        
        chosen_label = sorted_indices[0] # Default to largest area
        
        # Search through components to find the largest one touching the top
        found_top_touch = False
        for idx in sorted_indices:
            if stats[idx, cv2.CC_STAT_TOP] == 0:
                chosen_label = idx
                found_top_touch = True
                break
        
        largest_filtered[ccl_labels == chosen_label] = 255
        noise_map[(ccl_labels > 0) & (ccl_labels != chosen_label)] = 255
        
    # Step 6: Capping
    capped_mask = largest_filtered.copy()
    y_indices, x_indices = np.where(largest_filtered == 255)
    
    if len(y_indices) > 0:
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        # Extract X-coordinates strictly for the top row (min_y)
        x_at_min_y = x_indices[y_indices == min_y]
        x_min_top, x_max_top = np.min(x_at_min_y), np.max(x_at_min_y)
        
        # Extract X-coordinates strictly for the bottom row (max_y)
        x_at_max_y = x_indices[y_indices == max_y]
        x_min_bot, x_max_bot = np.min(x_at_max_y), np.max(x_at_max_y)
        
        # Draw the caps ONLY between the true physical bounds of the jet on those specific rows
        cv2.line(capped_mask, (x_min_top, min_y), (x_max_top, min_y), 255, 2)
        cv2.line(capped_mask, (x_min_bot, max_y), (x_max_bot, max_y), 255, 2)
        
    # Step 7: Contour Closure
    hole_filled_target = np.zeros_like(capped_mask)
    contours, _ = cv2.findContours(capped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(hole_filled_target, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1)
        
    # Step 8A & 8B: Spatiotemporal Core Isolation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_mask = cv2.morphologyEx(hole_filled_target, cv2.MORPH_OPEN, kernel)
    
    num_l_open, labels_open, stats_open, _ = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)
    main_core = np.zeros_like(opened_mask)
    
    if num_l_open > 1:
        areas_open = stats_open[1:, cv2.CC_STAT_AREA]
        # Sort indices to check from largest area to smallest
        sorted_indices_open = np.argsort(areas_open)[::-1] + 1 
        
        chosen_core_label = sorted_indices_open[0] # Default to largest area
        
        # Re-apply Physical Constraint Prioritization (Ytop == 0)
        for idx in sorted_indices_open:
            if stats_open[idx, cv2.CC_STAT_TOP] == 0:
                chosen_core_label = idx
                break
                
        main_core[labels_open == chosen_core_label] = 255
        
    # Step 9: Fragments Detachment
    detached = cv2.absdiff(hole_filled_target, main_core)
    
    # Step 10: Ghost Verification
    current_fragments_mask = np.zeros_like(hole_filled_target)
    num_f, f_labels, f_stats, _ = cv2.connectedComponentsWithStats(detached, connectivity=8)
    for i in range(1, num_f):
        f_mask = (f_labels == i).astype(np.uint8) * 255
        is_fragment = False
        if prev_main_core is not None:
            if np.count_nonzero(cv2.bitwise_and(f_mask, prev_main_core)) < (FRAGMENT_OVERLAP_THRESHOLD * f_stats[i, cv2.CC_STAT_AREA]):
                is_fragment = True
        if persistent_fragment_mask is not None and not is_fragment:
            if np.count_nonzero(cv2.bitwise_and(f_mask, persistent_fragment_mask)) > 0: 
                is_fragment = True
        if is_fragment:
            current_fragments_mask = cv2.bitwise_or(current_fragments_mask, f_mask)
            
    # Step 11: Final Verified Mask
    finalized_result = cv2.bitwise_and(hole_filled_target, cv2.bitwise_not(current_fragments_mask))
    recovered_mask = cv2.dilate(finalized_result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # Step 12: Forward Tracking Update
    next_persistent_mask = cv2.dilate(current_fragments_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)

    # ---------------------------------------------------------
    # PHASE 7: FINAL RIGHTMOST BOUNDARY EXTRACTION (Topological)
    # ---------------------------------------------------------
    final_boundary_vis = np.zeros((h, w, 3), dtype=np.uint8)
    final_boundary_vis[recovered_mask > 0] = [40, 40, 40] 
    
    right_boundary_pts = []
    contours_ext, _ = cv2.findContours(recovered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours_ext:
        # 1. Grab the single continuous outer shape
        largest_contour = max(contours_ext, key=cv2.contourArea)
        c = largest_contour.squeeze()
        
        # Ensure contour is valid (at least 3 points, 2D array)
        if c.ndim == 2 and len(c) > 2:
            # 2. Find absolute Top and Bottom limits
            y_min, y_max = c[:, 1].min(), c[:, 1].max()
            
            # Find Top-Right Anchor
            top_candidates = np.where(c[:, 1] == y_min)[0]
            idx_top = top_candidates[np.argmax(c[top_candidates, 0])]
            
            # Find Bottom-Right Anchor
            bot_candidates = np.where(c[:, 1] == y_max)[0]
            idx_bot = bot_candidates[np.argmax(c[bot_candidates, 0])]
            
            # 3. Extract both paths between the anchors
            idx1, idx2 = min(idx_top, idx_bot), max(idx_top, idx_bot)
            path1 = c[idx1:idx2+1]
            path2 = np.concatenate((c[idx2:], c[:idx1+1]), axis=0)
            
            # 4. The right boundary is the path that does NOT trace the left edge
            if path1[:, 0].min() > path2[:, 0].min():
                best_path = path1
            else:
                best_path = path2
                
            # Ensure points flow top-to-bottom (increasing Y) for curvature math
            if best_path[0, 1] > best_path[-1, 1]:
                best_path = best_path[::-1]

            # Save the exact coordinates for mathematical processing
            right_boundary_pts = [(int(x), int(y)) for x, y in best_path]

            # 5. Draw only this exact clipped segment for visualization
            pts_array = np.array(best_path, np.int32).reshape((-1, 1, 2))
            cv2.polylines(final_boundary_vis, [pts_array], False, (255, 0, 255), 2)

    return binary_all, final_boundary_vis, hole_filled_target, recovered_mask, main_core, next_persistent_mask, cluster_color_bgr, noise_map, current_fragments_mask, right_boundary_pts

# ==========================================
# 3. PIPELINE PER FOLDER
# ==========================================
def run_pipeline(input_folder, bg_folder, output_base):
    preprocessed_folder = os.path.join(output_base, "Preprocessed_Inverted")
    average_bg_folder = os.path.join(output_base, "Average_BG") 
    mask_folder = os.path.join(output_base, "Masks")
    video_folder = os.path.join(output_base, "Video")
    comparison_folder = os.path.join(output_base, "Comparison") 
    gmm_analysis_folder = os.path.join(output_base, "GMM_Analysis")
    gradients_folder = os.path.join(output_base, "Gradients")
    particle_plots_folder = os.path.join(output_base, "Particle_Plots")
    fragments_comp_folder = os.path.join(output_base, "Fragments_Removed_Comparison")
    curvature_folder = os.path.join(output_base, "Curvature_Heatmaps")
    spatiotemporal_2d_folder = os.path.join(output_base, "Spatiotemporal_2D_Video")

    for folder in [preprocessed_folder, average_bg_folder, mask_folder, video_folder, comparison_folder, gmm_analysis_folder, 
                   gradients_folder, particle_plots_folder, fragments_comp_folder, curvature_folder, spatiotemporal_2d_folder]:
        os.makedirs(folder, exist_ok=True)

    target_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.tif')])[:1500]
    if not target_files:
        print(f"  [ERROR] No target TIFF files found in {input_folder}")
        return

    # To initialize the 2D Video bounds correctly, we quickly read the shape of the very first frame.
    init_img = load_16bit_raw(os.path.join(input_folder, target_files[0]))
    h_img, w_img = init_img.shape

    # =======================================================
    # COMPUTE & SAVE AVERAGE BG
    # =======================================================
    average_normalized_bg = compute_average_normalized_bg(bg_folder)
    if average_normalized_bg is not None:
        tiff.imwrite(os.path.join(average_bg_folder, "Average_Normalized_BG.tif"), average_normalized_bg)
    else:
        print("  [WARNING] Skipping BG subtraction (No BG data found).")
    
    # INITIALIZE 2D VIDEO ENVIRONMENT
    video_2d_path = os.path.join(spatiotemporal_2d_folder, f"{os.path.basename(input_folder)}_2D_Boundary_Evolution.mp4")
    writer_2d = None

    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    ax_2d.set_xlim([-w_img * 0.05, w_img * 1.05])
    ax_2d.set_ylim([h_img * 1.05, -h_img * 0.05])
    ax_2d.set_xlabel("X Coordinate (Pixels)")
    ax_2d.set_ylabel("Y Coordinate (Pixels)")
    ax_2d.set_title("2D Boundary Evolution", fontweight='bold')
    
    active_line = None

    # =======================================================
    # THE FRAME LOOP (GMM TRAINING HAPPENS HERE NOW)
    # =======================================================
    video_name = f"{os.path.basename(input_folder)}_Target_Extraction.mp4"
    video_path = os.path.join(video_folder, video_name)
    writer = None 
    
    prev_main_core = None 
    persistent_fragment_mask = None

    fig, ax = plt.subplots(figsize=(7, 5))
    
    for f_idx, f in enumerate(tqdm(target_files, desc="  Processing", leave=False, file=sys.stdout, dynamic_ncols=True)):
        img = load_16bit_raw(os.path.join(input_folder, f))
        if img is None: continue
        
        # 1. EXACT Local Normalization Frame-by-Frame (UNTOUCHED CORE DATA)
        raw_inv = min_max_invert_16bit(img)

        # 2. Save to Preprocessed_Inverted (WITH ISOLATED BG SUBTRACTION)
        if average_normalized_bg is not None:
            # Using absdiff is mathematically safer for 16-bit uints to prevent 0-clipping data loss
            bg_subtracted = cv2.absdiff(average_normalized_bg, raw_inv)
            tiff.imwrite(os.path.join(preprocessed_folder, f), bg_subtracted)
        else:
            tiff.imwrite(os.path.join(preprocessed_folder, f), raw_inv)

        # 3. DYNAMIC GMM TRAINING PER FRAME 
        curr_pixels = raw_inv.flatten().astype(np.float32).reshape(-1, 1)
        anchored_means = np.linspace(np.min(curr_pixels), np.max(curr_pixels), 6)[1:-1].reshape(-1, 1)
        
        # Train Master GMM
        global_gmm = GaussianMixture(n_components=4, means_init=anchored_means, random_state=42, max_iter=300).fit(curr_pixels)
        sorted_indices = np.argsort(global_gmm.means_.flatten()) 
        curr_preds = global_gmm.predict(curr_pixels)
        
        # Train Nested GMM on C2
        c2_mask = (curr_preds == sorted_indices[2])
        c2_pixels = curr_pixels[c2_mask].flatten()

        if len(c2_pixels) > 50:
            nested_gmm = GaussianMixture(n_components=2, random_state=42, max_iter=700).fit(c2_pixels.reshape(-1, 1))
            sub_means = np.sort(nested_gmm.means_.flatten())
            local_jet_threshold = sub_means[1] + SHIFT_VALUE
        else:
            # Fallback if the fringe zone is essentially non-existent in a frame
            local_jet_threshold = global_gmm.means_[sorted_indices[2]][0] + SHIFT_VALUE

        # 4. PASS TO 14-STEP ENGINE
        binary_all, boundaries_vis, mathematical_closure, final_mask, next_core, next_ghost, cluster_color_bgr, noise_map, fragments_removed_mask, right_boundary_pts = isolate_target_global(
            raw_inv, prev_main_core, persistent_fragment_mask, local_jet_threshold, global_gmm, sorted_indices
        )
        
        bool_mask = final_mask > 127
        tiff.imwrite(os.path.join(mask_folder, f), final_mask)
        
        prev_main_core = next_core
        persistent_fragment_mask = next_ghost

        # =======================================================
        # CUSTOM VISUALIZATION: SPATIOTEMPORAL 2D & CURVATURE
        # =======================================================
        if len(right_boundary_pts) > 0:
            xs = np.array([p[0] for p in right_boundary_pts], dtype=np.float64)
            ys = np.array([p[1] for p in right_boundary_pts], dtype=np.float64)
            
            # Render 2D Evolving Boundary Video
            if active_line is not None:
                active_line.set_color('gray')
                active_line.set_alpha(0.15)
                active_line.set_linewidth(0.5)
            
            line_objects = ax_2d.plot(xs, ys, color='cyan', alpha=1.0, linewidth=2.5)
            active_line = line_objects[0]
            
            fig_2d.canvas.draw()
            img_2d_rgba = np.asarray(fig_2d.canvas.buffer_rgba())
            img_2d_bgr = cv2.cvtColor(img_2d_rgba, cv2.COLOR_RGBA2BGR)
            
            if writer_2d is None:
                writer_2d = cv2.VideoWriter(video_2d_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_2d_bgr.shape[1], img_2d_bgr.shape[0]))
            writer_2d.write(img_2d_bgr)

            # Extract and Plot Curvature Heatmap
            if len(xs) > 11:
                # Smooth BOTH X and Y to eliminate the jagged pixel staircase
                xs_smooth = scipy.signal.savgol_filter(xs, window_length=11, polyorder=2)
                ys_smooth = scipy.signal.savgol_filter(ys, window_length=11, polyorder=2)
                
                # First derivatives with respect to the index
                dx = np.gradient(xs_smooth)
                dy = np.gradient(ys_smooth)
                
                # Second derivatives
                d2x = np.gradient(dx)
                d2y = np.gradient(dy)
                
                # Standard Parametric Curvature Formula: |dx*d2y - dy*d2x| / (dx^2 + dy^2)^(3/2)
                numerator = np.abs(dx * d2y - dy * d2x)
                denominator = (dx**2 + dy**2)**1.5
                
                # Safety catch: prevent division by zero if the curve completely stops
                denominator[denominator == 0] = 1e-6
                
                curvature = numerator / denominator
                
                fig_curve, ax_curve = plt.subplots(figsize=(6, 8))
                sc = ax_curve.scatter(xs_smooth, ys_smooth, c=curvature, cmap='jet', s=10)
                plt.colorbar(sc, ax=ax_curve, label='Curvature (κ)')
                
                ax_curve.set_ylim([h_img, 0])
                ax_curve.set_xlim([0, w_img])
                ax_curve.set_title(f"Boundary Curvature - Frame {f_idx}")
                
                fig_curve.savefig(os.path.join(curvature_folder, f.replace('.tif', '.png')), bbox_inches='tight')
                plt.close(fig_curve)

        # =======================================================
        # CUSTOM VISUALIZATION: PARTICLE PLOTS
        # =======================================================
        num_labels_noise, _, stats_noise, _ = cv2.connectedComponentsWithStats(noise_map, connectivity=8)
        areas = stats_noise[1:, cv2.CC_STAT_AREA]
        
        ax.clear()
        if len(areas) > 0:
            ax.hist(areas, bins=50, color='crimson', edgecolor='black', alpha=0.7)
        ax.set_title(f"Number of Particles vs Size - {f}")
        ax.set_xlabel("Particle Size (Pixels)")
        ax.set_ylabel("Frequency (Number of Particles)")
        ax.grid(axis='y', alpha=0.75)
        fig.savefig(os.path.join(particle_plots_folder, f.replace('.tif', '.png')), bbox_inches='tight')

        # =======================================================
        # CUSTOM VISUALIZATION: GRADIENT MAP
        # =======================================================
        grad_x = cv2.Sobel(raw_inv, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(raw_inv, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_gray = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(os.path.join(gradients_folder, f.replace('.tif', '.jpg')), grad_gray)

        # =======================================================
        # CUSTOM VISUALIZATION: 4-PANEL VIDEO
        # =======================================================
        vis_inv_raw = normalize_to_8bit(raw_inv)
        
        vis_extracted_jet = np.zeros_like(vis_inv_raw)
        vis_extracted_jet[bool_mask] = vis_inv_raw[bool_mask]

        p1 = apply_pro_panel_for_video(vis_inv_raw, "1. INVERTED RAW")
        p2 = apply_pro_panel_for_video(vis_extracted_jet, "2. EXTRACTED JET")
        p3 = apply_pro_panel_for_video(boundaries_vis, "3. FINAL RIGHTMOST BOUNDARY")
        p4 = apply_pro_panel_for_video(noise_map, "4. NOISE MAP")

        row1_video = np.hstack((p1, p2))
        row2_video = np.hstack((p3, p4))
        combined_video = np.vstack((row1_video, row2_video))
        
        if writer is None:
            final_h, final_w = combined_video.shape[:2]
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (final_w, final_h))

        writer.write(combined_video)
        cv2.imwrite(os.path.join(comparison_folder, f.replace('.tif', '.jpg')), combined_video)

        # =======================================================
        # CUSTOM VISUALIZATION: FRAGMENTS PANELS
        # =======================================================
        f_p3 = apply_pro_panel_for_video(fragments_removed_mask, "3. FRAGMENTS REMOVED")
        row1_fragments = np.hstack((p1, p2))
        row2_fragments = np.hstack((f_p3, p4))
        combined_fragments = np.vstack((row1_fragments, row2_fragments))
        cv2.imwrite(os.path.join(fragments_comp_folder, f.replace('.tif', '.jpg')), combined_fragments)

        # ========================================================
        # GMM ANALYSIS LEGEND 
        # ========================================================
        p_gmm = apply_pro_panel_for_video(cluster_color_bgr, "2. GMM 4 CLUSTERS (LOCALLY NORMALIZED)")
        gmm_combined = np.hstack((p1, p_gmm))
        
        legend_height = 140
        legend_bgr = np.zeros((legend_height, gmm_combined.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(legend_bgr, (0, 0), (gmm_combined.shape[1], legend_height), (18, 18, 20), -1) 
        
        cv2.putText(legend_bgr, "GMM CLUSTER MAPPING (Locally Normalized Variance)", 
                    (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(legend_bgr, "Each cluster's intensity is stretched across its gradient to reveal hidden structures.", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        num_clusters = 4
        gap = 15
        total_gaps = gap * (num_clusters - 1)
        bar_x_start = 20
        bar_x_end = gmm_combined.shape[1] - 20
        available_width = (bar_x_end - bar_x_start) - total_gaps
        segment_width = available_width // num_clusters
        
        bar_y_start = 75
        bar_y_end = 95
        
        labels = ["C0: Thin Mist (Red)", "C1: Dense Mist (Purple)", "C2: Fringe (Yellow)", "C3: Jet Core (Green)"]
        
        palettes = {
            0: ((20, 20, 150), (100, 100, 255)), 
            1: ((150, 20, 150), (255, 100, 255)),
            2: ((20, 150, 150), (100, 255, 255)),
            3: ((20, 150, 20), (100, 255, 100))  
        }
        
        for i in range(num_clusters):
            start_x = bar_x_start + i * (segment_width + gap)
            end_x = start_x + segment_width
            
            dark = np.array(palettes[i][0], dtype=np.float32)
            light = np.array(palettes[i][1], dtype=np.float32)
            
            grad_img = np.zeros((bar_y_end - bar_y_start, segment_width, 3), dtype=np.float32)
            t = np.linspace(0, 1, segment_width)
            
            for row in range(grad_img.shape[0]):
                grad_img[row, :, 0] = dark[0] + (light[0] - dark[0]) * t
                grad_img[row, :, 1] = dark[1] + (light[1] - dark[1]) * t
                grad_img[row, :, 2] = dark[2] + (light[2] - dark[2]) * t
                
            legend_bgr[bar_y_start:bar_y_end, start_x:end_x] = grad_img.astype(np.uint8)
            cv2.rectangle(legend_bgr, (start_x, bar_y_start), (end_x, bar_y_end), (200, 200, 200), 1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(labels[i], font, 0.45, 1)[0]
            
            text_x = start_x + (segment_width - text_size[0]) // 2
            if text_size[0] > segment_width:
                text_x = start_x + 2 
            
            cv2.putText(legend_bgr, labels[i], (text_x, bar_y_end + 20), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(legend_bgr, "Low", (start_x, bar_y_end + 35), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
            
            high_size = cv2.getTextSize("High", font, 0.35, 1)[0]
            cv2.putText(legend_bgr, "High", (end_x - high_size[0], bar_y_end + 35), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        final_gmm_output = np.vstack((gmm_combined, legend_bgr))
        cv2.imwrite(os.path.join(gmm_analysis_folder, f.replace('.tif', '_GMM.jpg')), final_gmm_output)

    plt.close(fig)
    plt.close(fig_2d) 
    if writer is not None:
        writer.release()
    if writer_2d is not None:
        writer_2d.release()
    print(f"  Completed. Results saved in: {output_base}")

def batch_process():
    if not os.path.exists(BASE_ACTUAL_DIR):
        print(f"\n[FATAL ERROR] Check base folder path: \nActual: {BASE_ACTUAL_DIR}")
        return

    actual_folders = [f for f in os.listdir(BASE_ACTUAL_DIR) if os.path.isdir(os.path.join(BASE_ACTUAL_DIR, f))]

    for actual_name in actual_folders:
        input_folder = os.path.join(BASE_ACTUAL_DIR, actual_name)
        
        # Dynamically map to the corresponding BG folder by searching for the prefix
        # rsplit('_', 1)[0] cuts off the '_C001H001S0001' part to isolate your prefix
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
