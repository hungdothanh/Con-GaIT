

from src.data import preprocess_chunk
from src.model import build_cnn_lstm_model
from src.config import WINDOW_SIZE, NUM_CLASSES, PRETRAINED_MODEL_PATH, SELECTED_SENSORS




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
    