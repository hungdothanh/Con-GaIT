

#-----------inference.py------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from config import PRETRAINED_MODEL_PATH, SEGMENT_LENGTH, CLASS_NAMES, SENSOR_NAMES
from src.data import load_data, preprocess_file
from src.model import ParkinsonsGaitCNN
from src.lrp import LRPExplainer

# initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParkinsonsGaitCNN(input_channels=16, sequence_length=SEGMENT_LENGTH)
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
model.to(device).eval()
explainer = LRPExplainer(model, device)


def classification_fn(patient_name):
    data = load_data()[patient_name]
    segments = preprocess_file(data['gait_file'], SEGMENT_LENGTH)
    num_segments = len(segments)
    probs = []
    preds = []
    for seg in segments:
        inp = torch.FloatTensor(seg).unsqueeze(0).to(device)
        with torch.no_grad(): 
            out = model(inp)
            p = F.softmax(out,dim=1).cpu().numpy()[0]
            preds.append(p.argmax())
            probs.append(p.tolist())
    majority = int(np.bincount(preds).argmax())
    initial = probs[0]
    prob_dict = {CLASS_NAMES[i]: float(initial[i]) for i in range(4)}
    return CLASS_NAMES[majority], probs, prob_dict, num_segments-1


def get_top_features(segments, idx):
    relevance = explainer.explain_sample(segments[idx])
    rel = relevance.squeeze(0)
    scores = np.sum(np.abs(rel),axis=1)
    top5 = np.argsort(scores)[-5:][::-1]
    return [SENSOR_NAMES[int(i)] for i in top5]


#-------------------------------------
#---------Plot (Tab 3)----------------
#-------------------------------------
def plot_explanation(seg_idx, seg, sensor):
    relevance = explainer.explain_sample(seg)
    rel = relevance.squeeze(0)

    idx = SENSOR_NAMES.index(sensor)
    signal = seg[idx]
    rel = rel[idx]
    signal = signal.T
    rel = rel.T

    sampling_rate = 100  # Assuming 100Hz sampling rate
    segment_seconds = 10  # 10 seconds segment
    offset = seg_idx * segment_seconds
    time = np.arange(len(signal)) / sampling_rate + offset

    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.plot(time, signal, label=sensor)

    top_times = np.argsort(np.abs(rel))[-20:]
    for t in top_times:
        start = (t - 1) / sampling_rate + offset
        end = (t + 1) / sampling_rate + offset
        ax.axvspan(start, end, color='orange', alpha=0.4)
        # ax.axvspan(t-1, t+1, color='orange', alpha=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.legend(loc='upper right')
    ax.grid(True)
    return fig


def plot_full_segment_heatmap(seg_idx, seg):
    relevance = explainer.explain_sample(seg)
    rel = relevance.squeeze(0)
    n_sensors, n_samples = rel.shape
    # seg = seg.T
    # rel = rel.T

    sampling_rate = 100  # Assuming 100Hz sampling rate
    segment_seconds = 10  # 10 seconds segment
    offset = seg_idx * segment_seconds
    total_time = n_samples / sampling_rate

    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(rel, cmap='RdBu_r', aspect='auto', extent=[0 + offset, total_time + offset, n_sensors, 0])
    # ax.set_title('Full‐Segment LRP Heatmap')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensors')
    ax.set_yticks(range(len(SENSOR_NAMES)))
    ax.set_yticklabels(SENSOR_NAMES, fontsize=8)
    fig.colorbar(im, ax=ax, orientation='vertical', label='Relevance')
    return fig
