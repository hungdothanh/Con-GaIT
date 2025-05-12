
# Paths configuration
DATA_DIR = 'C:\\Users\\hungd\\OneDrive\\FAU\\StudOn\\SS25\\Human Computer Interaction (HCI)\\Student Research Competition (SRC)\\xai-based-pd-gait-severity\\data'

CSV_FILE = 'C:\\Users\\hungd\\OneDrive\\FAU\\StudOn\\SS25\\Human Computer Interaction (HCI)\\Student Research Competition (SRC)\\xai-based-pd-gait-severity\\label\\demographics.xls' 
PRETRAINED_MODEL_PATH = 'C:\\Users\\hungd\\OneDrive\\FAU\\StudOn\\SS25\\Human Computer Interaction (HCI)\\Student Research Competition (SRC)\\xai-based-pd-gait-severity\\weight\\saved_weights_200e.h5'
SELECTED_SENSORS = [1, 3, 7, 8, 9, 11, 15, 16]  # L1,L3,L7,L8,R1,R3,R7,R8 (0-indexed)
WINDOW_SIZE = 900  # 5 seconds at 100Hz as per paper
NUM_CLASSES = 4  # Healthy, Stage 2, Stage 2.5, Stage 3