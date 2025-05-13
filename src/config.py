
# Paths configuration
DATA_DIR = 'C:\\Users\\hungd\\OneDrive\\FAU\\StudOn\\SS25\\Human Computer Interaction (HCI)\\Student Research Competition (SRC)\\xai-based-pd-gait-severity\\data'

CSV_FILE = 'C:\\Users\\hungd\\OneDrive\\FAU\\StudOn\\SS25\\Human Computer Interaction (HCI)\\Student Research Competition (SRC)\\xai-based-pd-gait-severity\\label\\demographics.xls' 
PRETRAINED_MODEL_PATH = 'C:\\Users\\hungd\\OneDrive\\FAU\\StudOn\\SS25\\Human Computer Interaction (HCI)\\Student Research Competition (SRC)\\xai-based-pd-gait-severity\\weight\\saved_weights_200e.h5'
SELECTED_SENSORS = [1, 3, 7, 8, 9, 11, 15, 16]  # L1,L3,L7,L8,R1,R3,R7,R8 (0-indexed)
WINDOW_SIZE = 900  # 5 seconds at 100Hz as per paper
NUM_CLASSES = 4  # Healthy, Stage 2, Stage 2.5, Stage 3

css = """
.container {
    max-width: 900px;
    margin: auto;
}

/* Reduce vertical spacing between rows */
.gr-row {
    margin-top: 0px;
    margin-bottom: 0px;
}

h1 {
    text-align: center;
    display:block;
    font-size: 30px;
    font-weight: 900;
}


label, .block-label{
    font-weight: 790;
    font-size: 13px;
}

select, option{
    font-size: 16px;
    font-weight: 800;
}

textarea {
    font-size: 14px;
    font-weight: 500;
}

/* tab text */
button {
    font-size: 14px;
    font-weight: 550;
}

.gr-button, .gr-textbox, .gr-dropdown {
    font-size: 16px;
    font-weight: 800;
}

/* Blue-themed container for gait parameter boxes */
.gait-param-box {
    background: #fff;
    border: 1px solid #bfdbfe;  /* soft blue border */
    border-radius: 12px;
    padding: 12px;
    margin-top: -8px;
    margin-left: -8px;
    margin-right: 6px;
    mArgin-bottom: -4px;
    width: 100%;
    box-shadow: 0 4px 10px rgba(161, 196, 208, 0.2);  /* subtle blue shadow */
}
"""

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

