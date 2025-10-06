import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# IMPORTANT: Make sure this path is correct for your system.
CSV_FILE_PATH = r'D:\Flame_tracking\Dataset\flame_analysis_data.csv'

OUTPUT_FILENAME = 'flame_normalized_plot.png'


def plot_normalized_trajectories(file_path):
    """
    Loads flame trajectory data, normalizes the time for each flame to a 0-1
    scale, and plots the original position data on a single graph.
    """
    all_flame_data = []
    try:
        with open(file_path, 'r') as f:
            next(f)  # Skip the header line
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(',')
                all_flame_data.append({
                    'id': int(parts[0]),
                    'start_time': float(parts[1]),
                    'end_time': float(parts[2]),
                    'positions': [float(p) for p in parts[3:] if p]
                })

    except FileNotFoundError:
        print(f"[ERROR] The file '{file_path}' was not found.")
        print("Please make sure the script and the CSV file are in the same directory, or provide the full path.")
        return
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Could not parse data in '{file_path}'. Details: {e}")
        return

    if not all_flame_data:
        print(f"[ERROR] No valid data was loaded from '{file_path}'.")
        return

    # --- Create a figure and axes for the plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Iterate over the data to normalize and plot each flame ---
    for flame_data in all_flame_data:
        flame_id = flame_data['id']
        positions_cm = np.array(flame_data['positions'])
        start_time_s = flame_data['start_time']
        end_time_s = flame_data['end_time']
        
        # Create the original time axis
        time_axis_s = np.linspace(start_time_s, end_time_s, num=len(positions_cm))

        # --- TIME NORMALIZATION LOGIC ---
        # Normalize Time Axis to 0-1 range
        # Formula: (current_time - start_time) / (total_duration)
        total_duration = end_time_s - start_time_s
        # Avoid division by zero if start and end time are the same
        if total_duration > 0:
            normalized_time = (time_axis_s - start_time_s) / total_duration
        else:
            normalized_time = np.zeros_like(time_axis_s)


        # Plot original position vs. normalized time
        ax.plot(normalized_time, positions_cm, marker='.', linestyle='-', label=f'Flame {flame_id}')

    # --- Formatting the Plot ---
    ax.set_title('Flame Position over Normalized Time', fontsize=16)
    ax.set_xlabel('Normalized Time (Arbitrary Units)', fontsize=12)
    ax.set_ylabel('X-Position of Flame Centroid (cm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Flame ID')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # --- Save the figure to a file before showing it ---
    try:
        # dpi=300 sets a high resolution for the image.
        # bbox_inches='tight' ensures the saved image includes the legend.
        plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    except Exception as e:
        print(f"[ERROR] Could not save the plot. Reason: {e}")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_normalized_trajectories(CSV_FILE_PATH)

