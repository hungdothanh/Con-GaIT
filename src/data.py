#-----------------data.py-----------------
import yaml
import numpy as np
import pandas as pd


def preprocess_file(file_path, sequence_length=1000):
    """Preprocess a single gait file and return segments"""
    try:
        # Load data (assuming tab-separated values)
        data = pd.read_csv(file_path, sep='\t', header=None)

        if data.shape[1] > 1:
            features = data.iloc[:, 1:-2].values  # Skip time column
        else:
            features = data.values
        
        # Create segments of desired length
        segments = []
        if len(features) >= sequence_length:
            # Create overlapping segments
            step_size = sequence_length  # 50% overlap
            for i in range(0, len(features) - sequence_length + 1, step_size):
                segment = features[i:i+sequence_length]
                
                # Reshape for CNN input (channels, length)
                segment = segment.T  # Shape: (n_features, sequence_length)
                segments.append(segment)
        else:
            # Pad if too short
            padding = np.zeros((sequence_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])

            
            # Reshape for CNN input (channels, length)
            features = features.T  # Shape: (n_features, sequence_length)
            segments.append(features)
        
        return segments
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    

# Load patients data from YAML
def load_data():
    with open("src\data.yaml") as f:
        return yaml.safe_load(f)


def update_patient_info(patients_data, patient_name):
    if not patient_name or patient_name not in patients_data:
        return "", "", None
    pdata = patients_data[patient_name]
    personal = f"Age: {pdata['age']}               Gender: {pdata['gender']}\n"
    personal += f"Weight: {pdata['weight']}    Height: {pdata['height']}"
    return personal, 0, None  # reset segment idx to 0 and clear plot




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

def render_gait_parameter(value, param_type, min_val, max_val,
                          threshold, is_higher_better=True):
    """
    Create HTML visualization for gait parameters.

    If `value` is None, we draw the bar + threshold but suppress marker + number.
    """
    # Units lookup
    units = get_units(param_type)

    # Handle missing value
    if value is None:
        percent = 0
        display_val = "--"
        # neutral grey for pending
        bar_color = "#bbbbbb"
        threshold_percent = ((threshold - min_val) / (max_val - min_val) * 100)
    else:
        # Calculate percent
        percent = min(max(((value - min_val) / (max_val - min_val) * 100), 0), 100)
        threshold_percent = ((threshold - min_val) / (max_val - min_val) * 100)
        # green / red logic
        green_color = "#88c9bf"
        red_color   = "#d77c7c"
        if is_higher_better:
            bar_color = green_color if percent >= threshold_percent else red_color
        else:
            # for metrics where lower is better, invert coloring
            bar_color = red_color if percent >= threshold_percent else green_color
        display_val = f"{value:.1f}"  # when present

    # Build HTML
    html = f"""
    <div class="gait-param-box">
      <div style="font-size:15px; color:#1976d2; margin-bottom:8px; font-weight:700;">
        {param_type}
      </div>
      <div style="display:flex; justify-content:space-between; font-size:12px; color:#1976d2;">
        <span>{min_val} {units}</span>
        <span>{max_val} {units}</span>
      </div>
      <div style="position:relative; height:7px; margin:22px 0 14px 0; background:#f0f0f0; border:1px solid {bar_color}; border-radius:6px;">
        <!-- Threshold label -->
        <div style="position:absolute; left:{threshold_percent}%; top:-30px; transform:translateX(-50%); font-size:12px; color:#000;">
          {threshold} {units}
        </div>

        <!-- Filled portion -->
        <div style="position:absolute; left:0; width:{percent}%; height:100%; background:{bar_color}; border-radius:6px 0 0 6px;"></div>
        
        <!-- Threshold marker -->
        <div style="position:absolute; left:{threshold_percent}%; top:-6px; width:2px; height:18px; background:#000;"></div>
        
        {"<!-- Value marker -->" if value is None else f'<div style="position:absolute; left:{percent}%; top:-1px; width:2px; height:7px; background:#1976d2; border-radius:1px;"></div>'}
      </div>
      <div style="font-size:15px; color:{bar_color}; text-align:center; font-weight:bold;">
        {display_val} {units}
      </div>
    </div>
    """
    return html


#----------



