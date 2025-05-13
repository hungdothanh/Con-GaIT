
import gradio as gr
from PIL import Image
from config import css, js_func
from data import render_gait_parameter
from data import load_gait_data, visualize_gait_data, example_files
from utils import cls_predict
from xai import xai_predict_fn



def main():
    with gr.Blocks(title="GUI", js=js_func, css=css, theme='shivi/calm_seafoam').queue() as demo:
        # Logo
        gr.HTML("""
        <div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 10px;">
            <img src="https://www.tf.fau.de/files/2022/10/FAU-Logo.png" style="height: 80px;">
            <img src="https://www.mad.tf.fau.de/files/2024/02/rz1_MaDLogo_interim_CMYK.png" style="height: 55px;">
        </div>
        """)

        gr.Markdown("""# AI-supported PD Gait Analysis and Medication Monitoring""")
        data_state = gr.State()
        with gr.Row():

            '''
            Tab 1: Patient's  Data Overview
            '''
            with gr.Tab(label="Data Overview"):
                # Personal info AND GAIT DATA
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            select_patient_dropdown = gr.Dropdown(choices=["Patient 1", "Patient 2", "Patient 3", "Patient 4", "Patient 5", "Patient 6"], 
                                                                    label="Select Patient", interactive=True)
                        with gr.Row():
                            personal_info = "Age: 69                   Gender: Male  \nWeight: 69 kg         Height: 1.6 m"
                            patient_info_text = gr.Textbox(label="Patient's Info", value=personal_info)

                        # Patient's Info of Medical History
                        with gr.Row():
                            medication_info_text = gr.Textbox(label="Current Medication - Dosage", value="Levodopa  -  25/100 mg, 3 times/day", 
                                                              lines=1, interactive=False)                            
                        
                        # Checkbox to select which sensor to visualize
                        with gr.Row():
                            toggle_dropdown = gr.Dropdown(
                                choices=["Toggle Sensor Value", "Left Foot Total", "Right Foot Total", "8 optimal sensors"], 
                                label="Feature Selection", interactive=True, value=None
                            )

                            def toggle_checkbox_visibility(toggle_choice):
                                return gr.update(visible=(toggle_choice == "Toggle Sensor Value"))
                            # Link dropdown change to visibility function

                    with gr.Column(scale=3.5):
                        with gr.Row():
                            gait_img = Image.open('./figures/gait-data.jpg')
                            gait_img_plot = gr.Image(label="Gait Data Visualization", value=gait_img)
                            # gait_plot = gr.Plot(label="Gait Data Visualization")
                
                # Checkbox group for sensor selection
                with gr.Row():
                    # Sensor checkbox group, initially hidden
                    sensor_checkbox = gr.CheckboxGroup(
                        choices=[
                            "Left VGRF-1", "Left VGRF-2", "Left VGRF-3", "Left VGRF-4", 
                            "Left VGRF-5", "Left VGRF-6", "Left VGRF-7", "Left VGRF-8",
                            "Right VGRF-1", "Right VGRF-2", "Right VGRF-3", "Right VGRF-4", 
                            "Right VGRF-5", "Right VGRF-6", "Right VGRF-7", "Right VGRF-8",
                        ], 
                        label="Select Sensors to Visualize", visible=False, interactive=True, show_label=False
                    )
                    toggle_dropdown.change(
                            fn=toggle_checkbox_visibility, 
                            inputs=toggle_dropdown, 
                            outputs=sensor_checkbox
                        )

                # Patient's Gait Parameters
                with gr.Row():
                        with gr.Column(scale=1):
                            stride_amplitude = gr.HTML(
                                render_gait_parameter(value=45.0, param_type="STRIDE AMPLITUDE", 
                                                min_val=0, max_val=100, threshold=40, is_higher_better=True)
                            )
                        with gr.Column(scale=1):
                            stride_speed = gr.HTML(
                                render_gait_parameter(value=85.0, param_type="STRIDE SPEED", 
                                                min_val=40, max_val=140, threshold=90, is_higher_better=True)
                            )
                        with gr.Column(scale=1):
                            freezing_gait = gr.HTML(
                                render_gait_parameter(value=1.5, param_type="FREEZING OF GAIT", 
                                                min_val=0, max_val=10, threshold=3, is_higher_better=False)
                            )
                        with gr.Column(scale=1):
                            foot_lift = gr.HTML(
                                render_gait_parameter(value=8.5, param_type="HEIGHT OF FOOT LIFT", 
                                                min_val=0, max_val=20, threshold=7, is_higher_better=True)
                            )
                        with gr.Column(scale=1):
                            heel_strike = gr.HTML(
                                render_gait_parameter(value=25.0, param_type="HEEL STRIKE", 
                                                min_val=0, max_val=100, threshold=30, is_higher_better=False)
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("")
            
            """ 
            Tab 2: Medication Record
            """
            with gr.Tab(label="Medication Record"):
                with gr.Row():
                    with gr.Column(min_width=220):
                        with gr.Row():
                            gait_param_select_dropdown = gr.Dropdown(choices=["STRIDE AMPLITUDE", "STRIDE SPEED", "HEIGHT OF FOOT LIFT", "HEEL STRIKE", "FREEZING OF GAIT"],
                                                                    label="Select Gait Parameter", 
                                                                    value='STRIDE AMPLITUDE',
                                                                    interactive=True)
                    with gr.Column(min_width=220):
                        month_dropdown = gr.Dropdown(choices=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                                                    label="Select Month",
                                                    value='May',
                                                    interactive=True)
                    with gr.Column(min_width=220):
                        # Select time period (in Day) for 30 days
                        start_day_dropdown = gr.Dropdown(choices=[f"Day {i}" for i in range(1, 31)],
                                                        label="Start Day",
                                                        value='Day 1',
                                                        interactive=True)
                    with gr.Column(min_width=220):
                        end_day_dropdown = gr.Dropdown(choices=[f"Day {i}" for i in range(1, 31)],
                                                        label="End Day",
                                                        value='Day 15',
                                                        interactive=True)             

                with gr.Row():
                    medical_img = Image.open('./figures/medical.png')
                    medical_img_plot = gr.Image(label="Medication Tracking", value=medical_img)
                    # medication_history_plot = gr.Plot(label="Medication History Plot")

            """ 
            Tab 3: Predictions and Explanations
            """
            with gr.Tab(label="Predictions and Explanations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Severity Score
                        with gr.Row():
                            with gr.Row():
                                cls_button = gr.Button(value="Run Classification", size="md", interactive=True)
                            with gr.Row():
                                with gr.Column(min_width=120):
                                    hny_score = gr.Label(label="Hoehn & Yahr", value=4)
                                with gr.Column(min_width=120):
                                    updrs_score = gr.Label(label="MDS-UPDRS", value=3)
                            with gr.Row():
                                    with gr.Row(max_height=150):
                                        progression_probability = gr.Label(label="Progression Probability (%)", value=0.79)

                        # XAI and Textual Explanation
                        with gr.Row():
                            # Select Score to explain
                            xai_score_dropdown = gr.Dropdown(choices=["Hoehn and Yahr Score", "MDS-UPDRS Score", "Progression Probability"], 
                                                            label="Select Score to Explain",
                                                            value=None,
                                                            interactive=True)
                            xai_button = gr.Button(value="Run Explainable AI", size="md", interactive=True)

                    # Plotting area
                    with gr.Column(scale=3.5):
                        # XAI Plot for Hoehn and Yahr Score
                        with gr.Row():
                            xai_img = Image.open('./figures/xai2.png')
                            xai_hny_plot = gr.Image(label="Explanation for Selected Score", value=xai_img)
                            # xai_hny_plot = gr.Plot(label="Explanation for Hoehn and Yahr Score")
                        # # XAI Plot for MDS-UPDRS Score
                        # with gr.Row():
                        #     xai_updrs_plot = gr.Plot(label="Explanation for MDS-UPDRS Score")
                        # with gr.Row():
                        #     xai_progression_plot = gr.Plot(label="Explanation for Progression Probability")

                
                with gr.Row():
                    with gr.Column(scale=1.5):
                        with gr.Row():
                            xai_llm_text = gr.Textbox(label="Textual Explanation", lines=4,
                                                      placeholder="This is a textual explanation of the model's prediction. \n (e.g.) The model predicts a Hoehn and Yahr score of 4, indicating moderate to severe symptoms.")
                        with gr.Row():
                            xai_llm_button = gr.Button(value="Run Textual Explanation", size="md", interactive=True)
                    with gr.Column(scale=1):
                        with gr.Row():
                            give_feedback_text = gr.Textbox(label="Presciptions", lines=4,
                                                            placeholder="Please provide your feedback here \n (e.g.) adjust medication or physiotherapy plans")
                        with gr.Row():
                            give_feedback_button = gr.Button(value="Give Feedback", size="md", interactive=True)


    demo.launch()

if __name__ == "__main__":
    main()

