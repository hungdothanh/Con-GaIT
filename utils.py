
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.signal import periodogram
from scipy.stats import mode

from model import build_cnn_lstm_model
from config import WINDOW_SIZE, NUM_CLASSES, PRETRAINED_MODEL_PATH, SELECTED_SENSORS


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



def cls_predict(df, chunk_idx):
    """Run classification on a chunk of data"""
    model = build_cnn_lstm_model((WINDOW_SIZE, len(SELECTED_SENSORS)), NUM_CLASSES)
    model.load_weights(PRETRAINED_MODEL_PATH)
    
    # Preprocess the chunk
    processed_chunk = preprocess_chunk(df, chunk_idx)
    
    if processed_chunk is None:
        return "Chunk too short for processing."
    
    # Reshape for model input
    processed_chunk = processed_chunk.reshape(1, WINDOW_SIZE, len(SELECTED_SENSORS))
    
    # Predict
    predictions = model.predict(processed_chunk)

    # Round probabilities and format output string
    class_names = ["Healthy", "Stage 2", "Stage 2.5", "Stage 3"]
    predicted_probabilities = predictions[0]
    result_lines = [
        f"{name:<10} :    {prob:.3f}" for name, prob in zip(class_names, predicted_probabilities)
    ]

    
    return "\n".join(result_lines)
    