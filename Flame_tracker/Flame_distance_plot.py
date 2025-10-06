import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
CSV_FILE_PATH = r'D:\Flame_tracking\Dataset\flame_analysis_data.csv'

def plot_flame_trajectories(file_path):
    """
    Loads flame trajectory data from the specified CSV file, which has a variable
    number of columns per row, and generates a plot of position vs. time.
    """
    all_flame_data = []
    try:
        with open(file_path, 'r') as f:
            # Read the header but don't use it directly since the structure varies
            header = next(f).strip().split(',')
            
            for line in f:
                if not line.strip():  # Skip any empty lines
                    continue
                
                parts = line.strip().split(',')
                
                # Extract the first three fixed columns
                flame_id = int(parts[0])
                start_time = float(parts[1])
                end_time = float(parts[2])
                
                # The rest of the parts are the position measurements
                positions = [float(p) for p in parts[3:] if p]
                
                all_flame_data.append({
                    'id': flame_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'positions': positions
                })

    except FileNotFoundError:
        print(f"[ERROR] The file '{file_path}' was not found.")
        print("Please make sure this script is in the same directory as your CSV file, or provide the full path.")
        return
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Could not parse data in '{file_path}'. Please check the file for formatting errors.")
        print(f"Details: {e}")
        return

    if not all_flame_data:
        print(f"[ERROR] No valid data was loaded from '{file_path}'.")
        return

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Iterate over the processed data to plot each flame
    for flame_data in all_flame_data:
        flame_id = flame_data['id']
        positions_cm = flame_data['positions']
        start_time_s = flame_data['start_time']
        end_time_s = flame_data['end_time']
        
        # Create a precise time axis for this specific flame using its start and end times
        # np.linspace creates an array of evenly spaced numbers over a specified interval.
        if len(positions_cm) > 1:
            time_axis_s = np.linspace(start_time_s, end_time_s, num=len(positions_cm))
        else:
            # If there's only one data point, just use the start time
            time_axis_s = [start_time_s]

        # Plot position (y-axis) against the new time axis (x-axis)
        ax.plot(time_axis_s, positions_cm, marker='.', linestyle='-', label=f'Flame {flame_id}')

    # --- Formatting the Plot ---
    ax.set_title('Flame Position over Time', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('X-Position of Flame Centroid (cm)', fontsize=12)
    
    # Add a grid for easier reading of values
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a legend to identify each flame. Place it outside the plot area.
    ax.legend(title='Flame ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Adjust layout to prevent the legend from being cut off
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_flame_trajectories(CSV_FILE_PATH)
