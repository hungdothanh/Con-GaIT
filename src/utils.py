
# ----------------- src/utils.py (add this helper) -----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import SEGMENT_LENGTH


def convert_matplotlib_fig_to_base64(fig):
    from io import BytesIO
    import base64
    buf = BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


def get_sensor_index(sensor_name):
    # Map sensor names to dataframe column indices (0-based)
    num = int(sensor_name.split('-')[1])
    if sensor_name.startswith("Left"):
        return num
    else:
        return 8 + num
    
#-----------------------
#----Plot (Tab 1)-------
#-----------------------


def plot_gait_segment(patients_data, patient_name, sensor, start_idx):
    """
    Reads the gait file for patient, extracts a 10s segment starting at start_idx,
    and returns (new_start_idx, fig) where fig is a Matplotlib Figure.
    """
    pdata = patients_data.get(patient_name)
    if pdata is None or not sensor:
        return start_idx, None
    df = pd.read_csv(pdata["gait_file"], sep='\t', header=None)
    # Clip start
    start = max(0, min(start_idx, len(df) - SEGMENT_LENGTH))
    segment = df.iloc[start:start + SEGMENT_LENGTH]
    time = segment.iloc[:, 0]

    fig, ax = plt.subplots(figsize=(10, 3.9))

    if sensor == "Left Foot Total":
        y = segment.iloc[:, -2]
        ax.plot(time, y)
        ax.set_title("Left Foot Total")
    elif sensor == "Right Foot Total":
        y = segment.iloc[:, -1]
        ax.plot(time, y)
        ax.set_title("Right Foot Total")
    else:
        sensor_idx = get_sensor_index(sensor)
        y = segment.iloc[:, sensor_idx]
        ax.plot(time, y)
        ax.set_title(sensor)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.grid(True)

    return start, fig



#---------------------------
#--------Plot (Tab 2)-------
#---------------------------
def plot_medication_trend(patient_name,
                          daily_metrics: dict,
                          metric: str,
                          month: str,
                          start_day: int,
                          end_day: int,
                          med_start_map: dict = None):
    """
    Plot a sliced 30-day gait metric with fixed good/bad shading and med-start marker.
    """
    # --- Extract & slice the data ---
    arr = np.array(daily_metrics.get(metric, []))
    if arr.size != 30:
        raise ValueError(f"{metric} must have 30 values, got {arr.size}")
    s, e = start_day - 1, end_day - 1
    y = arr[s:e+1]
    x = np.arange(s + 1, e + 2)

    # --- Fixed bounds, thresholds, and directionality maps ---
    bounds = {
        "STRIDE AMPLITUDE": (0, 160),
        "STRIDE SPEED":     (40, 140),
        "HEEL STRIKE":      (0, 100)
    }
    units_map = {
        "STRIDE AMPLITUDE": "cm",
        "STRIDE SPEED":     "steps/min",
        "HEEL STRIKE":      "%"
    }
    thr_map = {
        "STRIDE AMPLITUDE": 100,
        "STRIDE SPEED":     100,
        "HEEL STRIKE":      60
    }
    higher_is_better = {
        "STRIDE AMPLITUDE": True,
        "STRIDE SPEED":     True,
        "HEEL STRIKE":      False
    }

    min_val, max_val = bounds[metric]
    thr = thr_map[metric]
    hib = higher_is_better[metric]

    # --- Colors ---
    good_color    = "#c7e8e1"  # very light seafoam green
    bad_color     = "#f8d7da"  # very light rose
    med_line_color = "#256d7b" # deep teal accent

    # --- Build the plot ---
    fig, ax = plt.subplots(figsize=(11, 4))

    if hib:
        # higher → better
        ax.axhspan(min_val, thr,      facecolor=bad_color,   alpha=0.4)
        ax.axhspan(thr,     max_val,  facecolor=good_color, alpha=0.4)
    else:
        # higher → worse
        ax.axhspan(min_val, thr,      facecolor=good_color, alpha=0.4)
        ax.axhspan(thr,     max_val,  facecolor=bad_color,   alpha=0.4)


    for med_name, mday in (med_start_map or {}).items():
        if start_day <= mday <= end_day:
            ax.axvline(mday, color=med_line_color, linestyle='-',
                    linewidth=2, label=f'{med_name} start (Day {mday})')

    # data + threshold line
    ax.plot(x, y, marker='o', linestyle='-', label=metric)
    ax.axhline(thr, color='#555555', linestyle='--', linewidth=1)

    # --- Labels and styling ---
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(f'Day in {month}')
    ax.set_ylabel(metric + f' ({units_map[metric]})')
    ax.set_title(f'{metric}: Day {start_day}–{end_day} in {month} for {patient_name}')
    ax.set_xticks(x)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.5)

    fig.tight_layout()
    return fig



def plot_metric_forecast(patient_name,
                         daily_metrics: dict,
                         forecast_metrics: dict,
                         metric: str,
                         month: str,
                         start_day: int,
                         end_day: int,
                         horizon_days: int,
                         med_start_map: dict = None):
    """
    Overlay observed (Day start–end) and forecast (next horizon_days)
    using fixed values from forecast_data[metric].
    """
    # --- historical slice (as before) ---
    arr = np.array(daily_metrics.get(metric, []))
    s, e = start_day - 1, end_day - 1
    obs = arr[s:e+1]
    x_obs = np.arange(s + 1, e + 2)

    # --- forecast slice from YAML ---
    fc_arr = np.array(forecast_metrics.get(metric, []))
    fc = fc_arr[1: 1 + horizon_days]
    x_fc = np.arange(e + 2, e + 2 + len(fc))

    x = np.concatenate([x_obs, x_fc])

    # --- reuse existing styling maps ---
    bounds   = {"STRIDE AMPLITUDE": (0,160), "STRIDE SPEED": (40,140), "HEEL STRIKE": (0,100)}
    thr_map  = {"STRIDE AMPLITUDE":100, "STRIDE SPEED":100, "HEEL STRIKE":60}
    hib      = {"STRIDE AMPLITUDE":True, "STRIDE SPEED":True, "HEEL STRIKE":False}[metric]
    units    = {"STRIDE AMPLITUDE":"cm", "STRIDE SPEED":"steps/min", "HEEL STRIKE":"%"}[metric]
    min_v, max_v = bounds[metric]; thr = thr_map[metric]

    good_bg = "#c7e8e1"; bad_bg = "#f8d7da"; med_col = "#256d7b"

    fig, ax = plt.subplots(figsize=(11,4))
    # background bands
    if hib:
        ax.axhspan(min_v, thr,      facecolor=bad_bg, alpha=0.4)
        ax.axhspan(thr,   max_v,    facecolor=good_bg, alpha=0.4)
    else:
        ax.axhspan(min_v, thr,      facecolor=good_bg, alpha=0.4)
        ax.axhspan(thr,   max_v,    facecolor=bad_bg, alpha=0.4)

    # med start markers
    for med,mday in (med_start_map or {}).items():
        if start_day <= mday <= end_day + horizon_days:
            ax.axvline(mday, color=med_col, linewidth=2, label=f'{med} start (Day {mday})')

    # plot observed
    ax.plot(x_obs, obs, marker='o', linestyle='-', label=f'Observed {metric}')
    # plot forecast
    ax.plot(x_fc, fc, marker='s', linestyle='--', label=f'Forecast next {horizon_days}d')


    # styling
    ax.axhline(thr, color='#555', linestyle='--')
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel(f'Day in {month}')
    ax.set_ylabel(f'{metric} ({units})')
    ax.set_title(f'{metric}: Day {start_day}–{end_day}+{horizon_days} forecast for {patient_name}')
    ax.set_xticks(x)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.5)
    fig.tight_layout()
    return fig


