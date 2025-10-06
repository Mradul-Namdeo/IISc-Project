import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
CSV_FILE_PATH = r'D:\Flame_tracking\Dataset\flame_analysis_data.csv'

def plot_flame_acceleration(file_path):
    """
    Loads flame trajectory data, calculates acceleration for each flame, and
    generates a simple subplot visualization of acceleration vs. time.
    """
    all_flame_data = []
    try:
        with open(file_path, 'r') as f:
            next(f)  # Skip header
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
        print("Please make sure the script and the CSV file are in the same directory.")
        return

    if not all_flame_data:
        print("[ERROR] No data loaded.")
        return

    num_flames = len(all_flame_data)

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(
        nrows=num_flames, 
        ncols=1, 
        figsize=(10, 2.5 * num_flames)
    )

    if num_flames == 1:
        axes = [axes]

    fig.suptitle('Flame Acceleration Over Time', fontsize=16, y=0.99)

    # Loop through each flame to calculate and plot acceleration
    for ax, flame_data in zip(axes, all_flame_data):
        flame_id = flame_data['id']
        positions_cm = np.array(flame_data['positions'])
        start_time_s = flame_data['start_time']
        end_time_s = flame_data['end_time']
        
        # We need at least 3 position points to get at least one acceleration value
        if len(positions_cm) < 3:
            ax.text(0.5, 0.5, 'Not enough data for acceleration', ha='center', va='center')
            continue

        time_axis_s = np.linspace(start_time_s, end_time_s, num=len(positions_cm))

        # --- Step 1: Calculate Velocity ---
        delta_position = np.diff(positions_cm)
        delta_time = np.diff(time_axis_s)
        velocity = delta_position / delta_time
        velocity_time_axis = (time_axis_s[:-1] + time_axis_s[1:]) / 2

        # --- Step 2: Calculate Acceleration ---
        delta_velocity = np.diff(velocity)
        delta_time_for_accel = np.diff(velocity_time_axis)
        acceleration = delta_velocity / delta_time_for_accel
        acceleration_time_axis = (velocity_time_axis[:-1] + velocity_time_axis[1:]) / 2

        # Plotting
        ax.plot(acceleration_time_axis, acceleration, marker='.', linestyle='-')
        
        ax.set_title(f'Flame {flame_id}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (cm/s²)')
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_filename = "flame_acceleration_subplots.png"
    plt.savefig(output_filename, dpi=120)
    print(f"Successfully created '{output_filename}'")


if __name__ == "__main__":
    plot_flame_acceleration(CSV_FILE_PATH)
