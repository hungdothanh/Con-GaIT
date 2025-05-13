
from lime import lime_tabular

from model import build_cnn_lstm_model
from utils import preprocess_chunk

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
to_categorical = tf.keras.utils.to_categorical
from config import WINDOW_SIZE, NUM_CLASSES, PRETRAINED_MODEL_PATH, SELECTED_SENSORS


data = np.load('preprocessed_data.npz')
X = data['X']
y = data['y']

# Convert labels to one-hot encoding
y_categorical = to_categorical(y, num_classes=NUM_CLASSES)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.1, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

test_model = build_cnn_lstm_model((WINDOW_SIZE, len(SELECTED_SENSORS)), NUM_CLASSES)
test_model.load_weights(PRETRAINED_MODEL_PATH)

def lime_predict(x):
    """Predict function for LIME"""
    x = x.reshape(-1, WINDOW_SIZE, len(SELECTED_SENSORS))
    pred = test_model.predict(x)
    return pred

def xai_predict_fn(df, chunk_idx, class_name):
    # Map class name to index
    class_to_idx_map = {
        "Healthy": 0,
        "Stage 2": 1,
        "Stage 2.5": 2,
        "Stage 3": 3
    }    

    class_idx = class_to_idx_map.get(class_name)
    print("Class index:", class_idx)
    if class_idx is None:
        return "Invalid class name."

    # Preprocess the chunk
    processed_chunk = preprocess_chunk(df, chunk_idx)
    print("Shape of processed chunk:", processed_chunk.shape)
    
    if processed_chunk is None:
        return "Chunk too short for processing."
    
    # Reshape for prediction
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    # reshape processed_chunk to 2D of shape (batch_size, WINDOW_SIZE * len(SELECTED_SENSORS))
    processed_chunk_flat = processed_chunk.reshape(1, WINDOW_SIZE * len(SELECTED_SENSORS))
    # processed_chunk_flat = processed_chunk.reshape(processed_chunk.shape[0], -1)
    print("Shape of flatten processed chunk:", processed_chunk_flat.shape)

    # Create a LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        mode="classification",
        feature_names=[f"t{t}_s{s}" for t in range(8) for s in range(8)],
        class_names=["Healthy", "Stage2", "Stage2.5", "Stage3"],
        discretize_continuous=False
    )

    print("Done creating explainer")

    # Generate explanation for the required class_idx (this will be the input chosen from the gui, and the LIME need to provide explanation and visualize it on the original df chunk)
    exp = explainer.explain_instance(
        data_row=processed_chunk_flat[0],
        predict_fn=lime_predict,
        num_features=8,
        top_labels=NUM_CLASSES,
        labels=[class_idx]
    )

    print("Done generating explanation")

    # Get the top features (indices and weights)
    lime_map = dict(exp.as_map()[exp.available_labels()[0]])  # {flat_index: weight, ...}

    # For each, convert flat index to (time, sensor)
    highlight_points = []
    for flat_idx, weight in lime_map.items():
        time_idx = flat_idx // 8
        sensor_idx = flat_idx % 8
        highlight_points.append((time_idx, sensor_idx, weight))

    print("Done getting highlight points")

    # original_df_chunk = df.iloc[chunk_idx * WINDOW_SIZE:(chunk_idx + 1) * WINDOW_SIZE, SELECTED_SENSORS]
    original_df_chunk = processed_chunk

    fig = plt.figure(figsize=(10,5))
    for sensor in range(len(SELECTED_SENSORS)):
        plt.plot(original_df_chunk[:, sensor], label=f"Sensor {SELECTED_SENSORS[sensor]}")
    
    # Highlight important points
    for time_idx, sensor_idx, weight in highlight_points:
        color = 'red' if weight > 0 else 'blue'
        plt.scatter(time_idx, original_df_chunk[time_idx, sensor_idx], color=color, s=60, edgecolor='k', zorder=5)

    plt.xlabel('Time Step')
    plt.ylabel('Sensor Value')
    plt.legend()

    return fig


    
