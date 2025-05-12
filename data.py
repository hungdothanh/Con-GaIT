import pandas as pd
import matplotlib.pyplot as plt
from config import WINDOW_SIZE, SELECTED_SENSORS

example_files = {
    "GaCo22_01": "data/GaCo22_01.txt", # 0
    "GaPt03_01": "data/GaPt03_01.txt", # 3
    "GaPt04_01": "data/GaPt04_01.txt", # 2
    "GaPt05_01": "data/GaPt05_01.txt", # 2
    "GaPt07_01": "data/GaPt07_01.txt", # 3
    "GaPt08_01": "data/GaPt08_01.txt", # 1
    "GaPt09_02": "data/GaPt09_02.txt", # 3
    "GaPt12_01": "data/GaPt12_01.txt", # 1
    "SiPt40_01": "data/SiPt40_01.txt", # 2
    "JuPt10_01": "data/JuPt10_01.txt", # 3
}


def load_gait_data(file):
    if hasattr(file, 'name'):  # uploaded file
        filepath = file.name
    elif isinstance(file, str):  # example file
        filepath = file
    else:
        return "Invalid file."

    try:
        # Load as whitespace-separated values, no headers
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
        if df.shape[1] != 19:
            return "Data format error: Expected 19 columns."

        return df
    except Exception as e:
        return f"Error loading file: {e}"



def visualize_gait_data(df, feature, chunk_idx):
    if isinstance(df, str):
        return df  # return error string

    start_idx = chunk_idx * WINDOW_SIZE
    end_idx = start_idx + WINDOW_SIZE
    df = df.iloc[start_idx:end_idx]

    time = df[0]
    fig, ax = plt.subplots(figsize=(10, 5))

    if feature == "Left Foot VGRF":
        for i in range(1, 9):
            ax.plot(time, df.iloc[:, i], label=f'Sensor {i}')
        ax.set_title(f'Left Foot VGRF Sensors (Chunk {chunk_idx})')
        ax.set_ylabel('Force (N)')

    elif feature == "Right Foot VGRF":
        for i in range(9, 17):
            ax.plot(time, df.iloc[:, i], label=f'Sensor {i}')
        ax.set_title(f'Right Foot VGRF Sensors (Chunk {chunk_idx})')
        ax.set_ylabel('Force (N)')

    elif feature == "Total Force":
        left_total = df[17]
        right_total = df[18]
        ax.plot(time, left_total, label='Left Total')
        ax.plot(time, right_total, label='Right Total')
        ax.set_title(f'Total Force (Chunk {chunk_idx})')
        ax.set_ylabel('Force (N)')
        ax.legend()
    
    elif feature == "8 Optimal Sensors":
        for i in range(1, 17):
            if i in SELECTED_SENSORS:
                ax.plot(time, df.iloc[:, i], label=f'Sensor {i}')
        ax.set_title(f'8 Optimal Sensors (Chunk {chunk_idx})')
        ax.set_ylabel('Force (N)')

    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.legend()
    return fig