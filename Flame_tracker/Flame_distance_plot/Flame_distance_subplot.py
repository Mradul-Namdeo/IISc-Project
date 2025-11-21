import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
CSV_FILE_PATH = r'D:\Flame_tracking\Dataset\flame_analysis_data.csv'

def create_simple_subplots(file_path):
    """
    Loads flame trajectory data and generates a simple, clean PNG image
    with a separate subplot for each flame. Each subplot includes x-axis ticks.
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
        return

    if not all_flame_data:
        print("[ERROR] No data loaded.")
        return

    num_flames = len(all_flame_data)

    # --- Create a figure and a grid of subplots ---
    # NOTE: We remove 'sharex=True' to allow each subplot to have its own x-ticks.
    fig, axes = plt.subplots(
        nrows=num_flames, 
        ncols=1, 
        figsize=(10, 2.5 * num_flames)
    )

    if num_flames == 1:
        axes = [axes]

    fig.suptitle('Flame Position Over Time', fontsize=16, y=0.99)

    # --- Loop through each flame and plot it on its own subplot ---
    for ax, flame_data in zip(axes, all_flame_data):
        flame_id = flame_data['id']
        positions_cm = flame_data['positions']
        start_time_s = flame_data['start_time']
        end_time_s = flame_data['end_time']
        time_axis_s = np.linspace(start_time_s, end_time_s, num=len(positions_cm))

        # Plot the data
        ax.plot(time_axis_s, positions_cm, marker='.', linestyle='-')

        # --- Customize each subplot ---
        ax.set_title(f'Flame {flame_id}')
        ax.set_xlabel('Time (s)')  # Add x-axis label to every subplot
        ax.set_ylabel('X-Position (cm)')
        ax.grid(True, linestyle='--', linewidth=0.5)

    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the final image
    output_filename = "flame_subplots.png"
    plt.savefig(output_filename, dpi=120)
    print(f"Successfully created '{output_filename}'")


if __name__ == "__main__":
    create_simple_subplots(CSV_FILE_PATH)
