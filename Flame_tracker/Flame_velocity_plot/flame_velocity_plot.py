import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
CSV_FILE_PATH = r'D:\Flame_tracking\Dataset\flame_analysis_data.csv'

def plot_flame_velocity(file_path):
    """
    Loads flame trajectory data, calculates the velocity for each flame, and
    generates a simple subplot visualization of velocity vs. time.
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

    fig.suptitle('Flame Velocity Over Time', fontsize=16, y=0.99)

    # Loop through each flame to calculate and plot velocity
    for ax, flame_data in zip(axes, all_flame_data):
        flame_id = flame_data['id']
        positions_cm = np.array(flame_data['positions'])
        start_time_s = flame_data['start_time']
        end_time_s = flame_data['end_time']
        
        if len(positions_cm) < 2:
            ax.text(0.5, 0.5, 'Not enough data to calculate velocity', ha='center', va='center')
            continue

        time_axis_s = np.linspace(start_time_s, end_time_s, num=len(positions_cm))

        # --- Velocity Calculation ---
        delta_position = np.diff(positions_cm)
        delta_time = np.diff(time_axis_s)
        velocity = delta_position / delta_time
        velocity_time_axis = (time_axis_s[:-1] + time_axis_s[1:]) / 2

        # Plotting
        ax.plot(velocity_time_axis, velocity, marker='.', linestyle='-')
        
        ax.set_title(f'Flame {flame_id}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (cm/s)')
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_filename = "flame_velocity_subplots.png"
    plt.savefig(output_filename, dpi=120)
    print(f"Successfully created '{output_filename}'")


if __name__ == "__main__":
    plot_flame_velocity(CSV_FILE_PATH)