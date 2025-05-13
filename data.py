import pandas as pd
import matplotlib.pyplot as plt
from config import WINDOW_SIZE, SELECTED_SENSORS
import numpy as np
import pandas as pd
from PyEMD import EMD

from scipy.signal import medfilt
from scipy.signal import periodogram
from scipy.stats import mode

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

# --------------- Demographics for Labeling -----------------
def load_demographics(csv_path):
    """Load demographics data with severity ratings"""
    df = pd.read_excel(csv_path)
    # Convert HoehnYahr to categorical values (0=healthy, 1=stage 2, 2=stage 2.5, 3=stage 3)
    severity_map = {
        0: 0,  # Healthy controls
        2.0: 1,    # Stage 2
        2.5: 2,    # Stage 2.5
        3.0: 3     # Stage 3
    }
    
    # Map severity from HoehnYahr column
    df['severity_class'] = df['HoehnYahr'].map(severity_map)
    return df


def extract_file_identifiers(filename):
    """Extract Study and Subject Number from filename like 'GaCo02_01.txt'"""
    # For filenames like GaCo02_01.txt
    parts = filename.split('_')[0]
    study = parts[:2]  # Extract study code (Ga)
    
    # Check if file is from control group or PD group
    if 'Co' in parts:
        group = 'CO'  # Control group
    else:
        group = 'PD'  # PD group
    
    # Extract subject number
    subjnum = int(parts[4:6] if len(parts) >= 6 else parts[2:4])
    
    return study, group, subjnum


# --------------- Preprocessing -----------------
def perform_emd_decomposition(signal):
    emd = EMD()
    imfs = emd(signal)
    if len(imfs) == 0:
        return signal
    # Power spectral analysis: select IMF with highest total power
    powers = []
    for imf in imfs:
        f, Pxx = periodogram(imf)
        powers.append(np.sum(Pxx))
    dominant_idx = np.argmax(powers)
    return imfs[dominant_idx]


def preprocess_file(file_path):
    """Process a single gait file with EMD decomposition"""
    # Load gait data
    data = pd.read_csv(file_path, sep='\t', header=None)
    
    # Extract selected sensors
    sensor_data = data.iloc[:, SELECTED_SENSORS].values
    sensor_data = medfilt(sensor_data, kernel_size=(3,1))
    
    # Skip first 10s and last 20s as per paper
    start_idx = 1000  # 10s * 100Hz
    end_idx = len(sensor_data) - 2000  # 20s * 100Hz
    if end_idx <= start_idx:
        return None  # File too short
    
    sensor_data = sensor_data[start_idx:end_idx, :]
    
    # Create segments
    segments = []
    for i in range(0, len(sensor_data) - WINDOW_SIZE, WINDOW_SIZE):
        segment = sensor_data[i:i + WINDOW_SIZE, :]
        if segment.shape[0] == WINDOW_SIZE:
            # Apply EMD to each channel
            imfs = []
            for ch in range(segment.shape[1]):
                imf = perform_emd_decomposition(segment[:, ch])
                imfs.append(imf)
            
            # Stack IMFs into a matrix
            processed_segment = np.column_stack(imfs)
            segments.append(processed_segment)
    
    if not segments:
        return None
    
    return np.array(segments)

def preprocess_chunk(df, chunk_idx):
    """Preprocess a chunk of data for classification"""
    start_idx = chunk_idx * WINDOW_SIZE
    end_idx = start_idx + WINDOW_SIZE
    df_chunk = df.iloc[start_idx:end_idx]

    sensor_data = df_chunk.iloc[:, SELECTED_SENSORS].values
    
    sensor_data = medfilt(sensor_data, kernel_size=(3,1))
    # print("Shape of chunk data:", sensor_data.shape)
    
    # Apply EMD to each channel
    imfs = []
    for ch in range(sensor_data.shape[1]):
        imf = perform_emd_decomposition(sensor_data[:, ch])
        imfs.append(imf)
    
    # Stack IMFs into a matrix
    processed_chunk = np.column_stack(imfs)
    
    return np.array(processed_chunk)



# --------------- Gait Parameters -----------------

def get_units(param_type):
    """Return appropriate units for each parameter type"""
    units_dict = {
        "STRIDE AMPLITUDE": "cm",
        "STRIDE SPEED": "steps/min",
        "HEIGHT OF FOOT LIFT": "cm",
        "HEEL STRIKE": "%",
        "FREEZING OF GAIT": "s"
    }
    return units_dict.get(param_type, "")


def render_gait_parameter(value, param_type, min_val, max_val, threshold, is_higher_better=True):
    """
    Create HTML visualization for gait parameters
    
    Parameters:
    - value: Current value of the parameter
    - param_type: Type of parameter (e.g., "STRIDE AMPLITUDE")
    - min_val: Minimum value for the parameter
    - max_val: Maximum value for the parameter
    - threshold: Threshold distinguishing good from bad
    - is_higher_better: If True, values above threshold are good; if False, values below threshold are good
    """
    # Calculate positions as percentages
    percent = min(max(((value - min_val) / (max_val - min_val) * 100), 0), 100)
    threshold_percent = ((threshold - min_val) / (max_val - min_val) * 100)
    green_color = "#88c9bf"  # green
    red_color = "#d77c7c"

    # Determine which side is good (green) based on is_higher_better
    if is_higher_better:
        if percent > threshold_percent:
            # If the value is above the threshold, set the color to green
            bar_color = green_color
        else:
            # If the value is below the threshold, set the color to red
            bar_color = red_color
    else:
        if percent > threshold_percent:
            # If the value is above the threshold, set the color to green
            bar_color = green_color
        else:
            # If the value is below the threshold, set the color to red
            bar_color = red_color
    
    units = get_units(param_type)

        # <div style="background:#fff; border-radius:12px; box-shadow:1 1px 4px #eee; padding:16px; margin-top:12px; width:370px;">
    html = f"""

    <div class="gait-param-box">
      <div style="font-size:15px; color:#1976d2; margin-bottom:6px; font-weight:700;">{param_type}</div>
      <div style="display:flex; justify-content:space-between; font-size:12px; color:#1976d2;">
        <span>{min_val} {units}</span>
        <span>{max_val} {units}</span>
      </div>
      <div style="position:relative; height:7px; margin:8px 0 15px 0; background:#f0f0f0; border:1px solid {bar_color}; border-radius:6px;">
        <!-- Threshold label -->
        <div style="position:absolute; left:{threshold_percent}%; top:-23px; transform:translateX(-50%); font-size:12px; color:#000;">
            <span> {threshold} {units} <span>
        </div>

        <!-- Left section -->
        <div style="position:absolute; left:0; width:{percent}%; height:100%; background:{bar_color}; border-radius:6px 0 0 6px;"></div>
        
        <!-- Threshold marker -->
        <div style="position:absolute; left:{threshold_percent}%; top:-6px; width:2px; height:18px; background:#000000;"></div>
        
        <!-- Value marker -->
        <div style="position:absolute; left:{percent}%; top:-1px; width:2px; height:7px; background:#1976d2; border-radius:1px;"></div>
      </div>
      <div style="font-size:15px; color:{bar_color}; text-align:center; font-weight:bold;">{value:.1f} {units}</div>
    </div>
    """
    return html
