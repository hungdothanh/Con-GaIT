#-----------------app.py-----------------
import numpy as np
import gradio as gr
from config import css, js_func, SEGMENT_LENGTH, CLASS_NAMES
from src.data import render_gait_parameter, load_data, update_patient_info, preprocess_file
from src.inference import classification_fn, get_top_features, plot_explanation, plot_full_segment_heatmap
from src.gpt_explain_gait import gpt_flag_justify
from src.utils import plot_gait_segment, plot_medication_trend, plot_metric_forecast




def main():
    with gr.Blocks(title="GUI", js=js_func, css=css, theme='shivi/calm_seafoam').queue() as demo:
        # Logo
        # gr.HTML("""
        # <div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 10px;">
        #     <img src="https://www.tf.fau.de/files/2022/10/FAU-Logo.png" style="height: 80px;">
        #     <img src="https://www.mad.tf.fau.de/files/2024/02/rz1_MaDLogo_interim_CMYK.png" style="height: 55px;">
        # </div>
        # """)

        gr.HTML("""
        <style>
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            /* remove the old margin, we’ll apply it on the icon itself */
        }
        /* style the icon itself */
        .tooltip .tooltip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.2em;
            height: 1.2em;
            margin-left: 8px;
            font-size: 1em;
            color: #888;
            font-weight: bold;
            border: 2px solid #888;       /* ← thicker outline */
            border-radius: 50%;           /* ← circle shape */
            line-height: 1;               /* center the “i” vertically */
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 220px;
            background-color: rgba(0,0,0,0.75);
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 100;
            bottom: 125%; /* position above the icon */
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 0.9em;
            line-height: 1.2em;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%; /* arrow at bottom of tooltip */
            left: 50%;
            margin-left: -5px;
            border-width: 7px;
            border-style: solid;
            border-color: rgba(0,0,0,0.75) transparent transparent transparent;
        }
        
        /* 1. Tell .tooltip-right that its popup should be to the right */
        .tooltip-right .tooltiptext {
        /* Remove the old “above” settings */
        bottom: auto;
        left: 125%;        /* push it just past the icon */
        top: 50%;          /* vertically center on the icon */
        transform: translateY(-50%);
        }

        /* 2. Flip the little arrow so it points back at the icon */
        .tooltip-right .tooltiptext::after {
        top: 50%;
        left: 0;           /* attach to the left edge of the popup */
        margin: 0;
        transform: translateY(-50%);
        border-color: transparent rgba(0,0,0,0.75) transparent transparent;
        }
        </style>
        """)



        gr.Markdown("""# Contestable Gait Interpretation & Tracking""")

        patients_data = load_data()
        forecast_flag = gr.State(0)  # 0 for history, 1 for forecast
        prob_state = gr.State([])
        feedback_list = []

        with gr.Row():

            '''
            Tab 1: Patient's  Data Overview
            '''
            with gr.Tab(label="Gait Session Summary"):
                # Personal info AND GAIT DATA
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            select_patient_dropdown = gr.Dropdown(
                                choices=list(patients_data.keys()),
                                label="Select Patient", interactive=True
                            )
                        with gr.Row():
                            patient_info_text = gr.Textbox(label="Patient's Info")
                        
                        
                        # Checkbox to select which sensor to visualize
                        with gr.Row():
                            sensor_dropdown = gr.Dropdown(
                                choices=[f"Left VGRF-{i}" for i in range(1,9)] + [f"Right VGRF-{i}" for i in range(1,9)] + ["Left Foot Total", "Right Foot Total"], 
                                label="Select Sensor", interactive=True
                            )
                        with gr.Row():
                            segment_slider = gr.Slider(
                                minimum=0, maximum=0, step=1, value=0, visible=True,
                                label="Select Segment", interactive=True
                            )

                    with gr.Column(scale=3.5): 
                        with gr.Row():
                            gait_plot = gr.Plot(label="Gait Data Visualization")


                # Markdown to Introduce the Gait Metrics Overview
                with gr.Row():
                    gr.Markdown("""
                    <div style="text-align: center; font-size: 22px; font-weight: 600; margin: 10px 0;">
                        Gait Metrics Overview
                    </div>
                    """, elem_id="gait-metrics-overview")

                # Patient's Gait Parameters
                with gr.Row():
                        with gr.Column(scale=1):
                            stride_amplitude = gr.HTML(
                                render_gait_parameter(value=None, param_type="STRIDE AMPLITUDE", 
                                                min_val=0, max_val=160, threshold=100, is_higher_better=True)
                            )
                        with gr.Column(scale=1):
                            stride_speed = gr.HTML(
                                render_gait_parameter(value=None, param_type="STRIDE SPEED", 
                                                min_val=40, max_val=140, threshold=100, is_higher_better=True)
                            )
                        with gr.Column(scale=1):
                            heel_strike = gr.HTML(
                                render_gait_parameter(value=None, param_type="HEEL STRIKE", 
                                                min_val=0, max_val=100, threshold=60, is_higher_better=False)
                            )
                        # with gr.Column(scale=1):
                        #     freezing_gait = gr.HTML(
                        #         render_gait_parameter(value=1.5, param_type="FREEZING OF GAIT", 
                        #                         min_val=0, max_val=10, threshold=3, is_higher_better=False)
                        #     )
                        # with gr.Column(scale=1):
                        #     foot_lift = gr.HTML(
                        #         render_gait_parameter(value=8.5, param_type="HEIGHT OF FOOT LIFT", 
                        #                         min_val=0, max_val=20, threshold=7, is_higher_better=True)
                        #     )
                        # with gr.Column(scale=1):
                        #     gr.Markdown("")
            
            """ 
            Tab 2: Medication Record
            """
            with gr.Tab(label="Treatment Trend View"):
                with gr.Row():
                    with gr.Column(min_width=220):
                        with gr.Row():
                            gait_param_select_dropdown = gr.Dropdown(choices=["STRIDE AMPLITUDE", "STRIDE SPEED", "HEEL STRIKE"],
                                                                    label="Select Gait Parameter", 
                                                                    value=None,
                                                                    interactive=True)
                    with gr.Column(min_width=220):
                        month_dropdown = gr.Dropdown(choices=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                                                    label="Select Month",
                                                    interactive=True)
                    with gr.Column(min_width=220):
                        # Select time period (in Day) for 30 days
                        start_day_dropdown = gr.Dropdown(choices=[f"Day {i}" for i in range(1, 31)],
                                                        label="Start Day",
                                                        interactive=True)
                    with gr.Column(min_width=220):
                        end_day_dropdown = gr.Dropdown(choices=[f"Day {i}" for i in range(1, 31)],
                                                        label="End Day",
                                                        interactive=True)             

                with gr.Row():
                    gr.Markdown("""
                        <div style="text-align: center; font-size: 22px; font-weight: 600; margin: 24px 0;">
                            Medication Trend Visualization
                        </div>
                        """, elem_id="medication-trend-visualization")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                            <div style="display:inline-flex; align-items:center; margin-bottom:-2px;">
                                <span style="font-size:16px; color:#0066cc; font-weight:500;">
                                Select Medications
                                </span>
                                <span class="tooltip">
                                <span class="tooltip-icon">i</span>
                                <span class="tooltiptext">
                                    Here you can pick which of the patient’s current and past
                                    medications to overlay on the gait‐metric plot. Each
                                    medication’s start day will be marked on the timeline.
                                </span>
                                </span>
                            </div>
                            """)
                        med_checkbox = gr.CheckboxGroup(
                            choices=[],        # populated on patient change
                            label='',
                            interactive=True
                        )
                                        
                        # New forecast controls appear after history rendered
                        with gr.Row():
                            horizon_slider = gr.Slider(
                                minimum=3, maximum=15, step=1, value=0,
                                label="Forecast Horizon (days)", interactive=True
                            )
                        with gr.Row():
                            predict_btn = gr.Button(
                                value="Predict Improvement", size="md", interactive=True
                            )
                    with gr.Column(scale=3.5):
                        medication_history_plot = gr.Plot(label='')



            """ 
            Tab 3: Predictions and Explanations
            """
            with gr.Tab(label="Predictive Insight and Explanation"):
                with gr.Row():
                    with gr.Column(scale=3.5):
                        with gr.Row():
                                with gr.Column(scale=3):
                                    with gr.Row():
                                        gr.Markdown("""
                                            <div style="text-align: center; font-size: 16px; font-weight: 600; margin: 24px 0;">

                                            </div>
                                            """)
                                    with gr.Row():
                                        cls_btn = gr.Button(value="Run Classification", size="md", interactive=True)
                                    with gr.Row():
                                        slider = gr.Slider(minimum=0, maximum=0, step=1, visible=True, label="Select Segment")
                                    # XAI
                                    with gr.Row():
                                        xai_btn = gr.Button(value="Run Explainable AI", size="md", interactive=True)
                                    with gr.Row():
                                        feat_dd = gr.Dropdown(label="Select Top Feature")
                                with gr.Column(scale=3):
                                    gr.HTML("""
                                        <div style="text-align: center; margin: 5px 0;">
                                            <span style="text-align: center; font-size:16px; color:#0066cc; font-weight:700;">
                                            Hoehn & Yahr Score
                                            </span>
                                            <span class="tooltip tooltip-right">
                                            <span class="tooltip-icon">i</span>
                                            <span class="tooltiptext">
                                                Shows the predicted probability for each of the four Hoehn & Yahr stages and highlights the stage with the highest probability.
                                            </span>
                                            </span>
                                        </div>
                                        """)

                                    # with gr.Column():
                                    #     flag_btn = gr.HTML(
                                    #                 '<span id="flag-icon" style="cursor:pointer; font-size:20px; margin-left:10px;">🚩</span>'
                                    #             )
                                    proba = gr.Label(label='')
                                with gr.Column(scale=1):
                                    with gr.Row():
                                        gr.Markdown("""
                                            <div style="text-align: center; font-size: 16px; font-weight: 600; margin: 24px 0;">

                                            </div>
                                            """)
                                    with gr.Row():
                                        flag_btn = gr.Button(
                                            value="🚩",
                                            elem_id="flag-icon",
                                            size="sm",
                                            variant="secondary"
                                        )
                                        flag_taxonomy = gr.CheckboxGroup(choices=["Factual Error", "Normative Conflict", "Reasoning Flaw"], label="")

                        # Plotting area
                        # with gr.Row():
                            # gr.Markdown("""
                            #     <div style="text-align: center; font-size: 16px; font-weight: 600; margin: 24px 0;">
                            #         Sensor-wise Explanation Overlay on Gait Segment
                            #     </div>
                            #     """, elem_id="explanation-overlay")
                        with gr.Row():
                            xai_hny_plot = gr.Plot(label="Explanation Overlay on Gait Segment")


                    with gr.Column(scale=1):
                        textual_btn = gr.Button(value="Run Textual Explanation", size="md", visible=False)
                        textual_output = gr.Textbox(label="Textual Explanation", lines=10, interactive=False, 
                                            placeholder=(
                                                "This area shows the auto-generated LLM justification for the selected gait segment. \n"
                                                "If you send feedback below, the model will revise or extend this justification accordingly."
                                            ),
                                            visible=False)
                        feedback_btn = gr.Button(value="Send Feedback", size="md", interactive=True, visible=False)
                        feed_back_output = gr.Textbox(label="Feedback", lines=5, interactive=True,
                                                      placeholder=( 
                                                            "Provide your clinical contestation and observations to challenge the AI’s output. \n"
                                                            "Your feedback will drive LLM debate for re-justification and inform model refinement."
                                                      ),
                                                      visible=False
                                                     )
                        # "Enter your clinical feedback here. For example:\n"
                        #     "• Which highlighted regions you disagree with and why\n"
                        #     "• Relevant patient/context notes the model may have missed\n"
                        #     "• Suggestions to improve future explanations"

                        # feedback_status = gr.Markdown("", label="", visible=True)

        # ------------------------------
        # Tab1: Patient selection resets info & plot
        # ------------------------------
        def on_patient_change(patient_name):
            # update info and slider range
            if not patient_name or patient_name not in patients_data:
                return "", "", gr.update(maximum=0, value=0, visible=True), None, "", "", "", []
            
            # personal and medication
            personal, _, _ = update_patient_info(patients_data, patient_name)

            meds_map = patients_data[patient_name]["medications"]
            med_names = list(meds_map.keys())
            med_checkbox_update = gr.update(choices=med_names, value=[], visible=True)
            # med_info_str = med_names[-1] if med_names else ""

            # compute max segments
            segments = preprocess_file(patients_data[patient_name]['gait_file'], SEGMENT_LENGTH)
            max_idx = len(segments) - 1
            slider_update = gr.update(maximum=max_idx, value=0, visible=True)

            # fetch last-day gait metrics
            dm = patients_data[patient_name].get("daily_metrics", {})
            amp_val = dm.get("STRIDE AMPLITUDE", [0])[-1]
            spd_val = dm.get("STRIDE SPEED", [0])[-1]
            hs_val  = dm.get("HEEL STRIKE",   [0])[-1]
    
            # render each param box
            html_amp = render_gait_parameter(
                value=amp_val, param_type="STRIDE AMPLITUDE",
                min_val=0,   max_val=160, 
                threshold=100, is_higher_better=True
            )
            html_spd = render_gait_parameter(
                value=spd_val, param_type="STRIDE SPEED",
                min_val=40,  max_val=140, 
                threshold=100, is_higher_better=True
            )
            html_hs = render_gait_parameter(
                value=hs_val,  param_type="HEEL STRIKE",
                min_val=0,     max_val=100,
                threshold=60,  is_higher_better=False
            )
    
            return personal, slider_update, None, html_amp, html_spd, html_hs, med_checkbox_update

        
        select_patient_dropdown.change(
            fn=on_patient_change,
            inputs=[select_patient_dropdown],  
            outputs=[patient_info_text, segment_slider, gait_plot, stride_amplitude, stride_speed, heel_strike, med_checkbox]
        )


        def on_sensor_dropdown_change(patient, sensor, seg_idx):
            start = seg_idx * SEGMENT_LENGTH
            start, fig = plot_gait_segment(patients_data, patient, sensor, start)
            return fig

        sensor_dropdown.change(
            fn=on_sensor_dropdown_change,
            inputs=[select_patient_dropdown, sensor_dropdown, segment_slider],
            outputs=[gait_plot]
        )

        # Plot on slider change
        def on_slider_change(patient, sensor, seg_idx):
            start = seg_idx * SEGMENT_LENGTH
            start, fig = plot_gait_segment(patients_data, patient, sensor, start)
            return fig

        segment_slider.change(
            fn=on_slider_change,
            inputs=[select_patient_dropdown, sensor_dropdown, segment_slider],
            outputs=[gait_plot]
        )

        
        # ----------------------------
        # Tab2: Medication Record
        # ----------------------------
        # hook Gradio controls directly to our helper in src/data.py
        def _med_trend_callback(patient_name, metric, month, start_day, end_day, selected_meds):
            # don't attempt to plot until everything is selected
            if not all([patient_name, metric, month, start_day, end_day]):
                return None, 0

            # now safe to split
            s = int(start_day.split()[1])
            e = int(end_day.split()[1])

            # Pull meds mapping
            meds_map = patients_data[patient_name]["medications"]
            # Pass the selected meds and their start days
            med_starts = {m: meds_map[m] for m in (selected_meds or [])}
            fig = plot_medication_trend(
                patient_name,
                patients_data[patient_name]["daily_metrics"],
                metric, month, s, e,
                med_start_map=med_starts
            )

            return fig, 0
        
        for ctrl in [gait_param_select_dropdown, month_dropdown, start_day_dropdown, end_day_dropdown]:
            ctrl.change(
                fn=_med_trend_callback,
                inputs=[select_patient_dropdown,
                         gait_param_select_dropdown,
                         month_dropdown,
                         start_day_dropdown,
                         end_day_dropdown,
                         med_checkbox],
                outputs=[medication_history_plot, forecast_flag]
            )

        def _forecast_callback(patient_name, metric, month, start_day, end_day, horizon, selected_meds):
            if not all([patient_name, metric, month, start_day, end_day]):
                return None, 0
            s = int(start_day.split()[1]); e = int(end_day.split()[1])

            meds_map = patients_data[patient_name]["medications"]
            # Pass the selected meds and their start days
            med_starts = {m: meds_map[m] for m in (selected_meds or [])}

            # call forecasting function
            fig = plot_metric_forecast(
                patient_name,
                patients_data[patient_name]["daily_metrics"],
                patients_data[patient_name]["forecast_metrics"],
                metric, month, s, e,
                horizon_days=horizon,
                med_start_map=med_starts
            )
            return fig, 1

        def med_checkbox_dispatch(
            patient_name, metric, month, start_day, end_day,
            forecast_flag_val, horizon, selected_meds
        ):
            # only plot history until the user has explicitly asked for a forecast
            if forecast_flag_val == 0:
                return _med_trend_callback(patient_name, metric, month, start_day, end_day, selected_meds)
            else:
                return _forecast_callback(patient_name, metric, month, start_day, end_day, horizon, selected_meds)

        med_checkbox.change(
            fn=med_checkbox_dispatch,
            inputs=[
                select_patient_dropdown,
                gait_param_select_dropdown,
                month_dropdown,
                start_day_dropdown,
                end_day_dropdown,
                forecast_flag,        # <-- gr.State value
                horizon_slider,       # always pass it, even if not used
                med_checkbox
            ],
            outputs=[medication_history_plot, forecast_flag]
        )


        predict_btn.click(
            fn=_forecast_callback,
            inputs=[select_patient_dropdown, gait_param_select_dropdown,
                    month_dropdown, start_day_dropdown, end_day_dropdown,
                    horizon_slider, med_checkbox],
            outputs=[medication_history_plot, forecast_flag]
        )

        # ----------------------------
        # Tab3: Predictions and Explanations
        # ----------------------------
        def run_and_configure(patient_name):
            _, probs, prob_dict, max_idx = classification_fn(patient_name)          
            slider_update = gr.update(maximum=max_idx, value=0, visible=True)
            return probs, prob_dict, slider_update
        cls_btn.click(
            fn=run_and_configure,
            inputs=[select_patient_dropdown],
            outputs=[prob_state, proba, slider]
        )
        slider.change(fn=lambda probs,i: ({CLASS_NAMES[j]:probs[i][j] for j in range(4)}),
                        inputs=[prob_state, slider], 
                        outputs=[proba]
        )

        def on_run_xai(patient_name, seg_idx):
            # 1) compute top‐5 sensors
            segments = preprocess_file(patients_data[patient_name]['gait_file'], SEGMENT_LENGTH)
            top5 = get_top_features(segments, seg_idx)
            # 2) plot full‐segment heatmap
            heatmap_fig = plot_full_segment_heatmap(seg_idx, segments[seg_idx])
            # 3) tell Gradio: update dropdown & give the heatmap fig
            return gr.update(choices=top5, value=top5[0])

        xai_btn.click(
            fn=on_run_xai,
            inputs=[select_patient_dropdown, slider],
            outputs=[feat_dd],
        )

        feat_dd.change(fn=lambda patient_name, i,f: plot_explanation(i, 
                                                        preprocess_file(patients_data[patient_name]['gait_file'], SEGMENT_LENGTH)[i], 
                                                        f),
                        inputs=[select_patient_dropdown, slider, feat_dd], 
                        outputs=[xai_hny_plot]
        )

        flag_btn.click(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            ),
            inputs=[],
            outputs=[textual_btn, textual_output, feedback_btn, feed_back_output]
        )

        # Wrapper that regenerates actual Figure objects, not Gradio placeholders
        def run_textual_explanation(patient_name, seg_idx, sensor, probs, flag_types):
            # Get the top-5 sensors
            segments = preprocess_file(patients_data[patient_name]['gait_file'], SEGMENT_LENGTH)
            top5 = get_top_features(segments, seg_idx)


            #  Build one LRP explanation figure for the selected sensor in the top-5
            fig = plot_explanation(seg_idx, segments[seg_idx], sensor)

            heatmap_fig = plot_full_segment_heatmap(seg_idx, segments[seg_idx])

            pred_stage = CLASS_NAMES[np.argmax(probs[seg_idx])]

            return gpt_flag_justify(
                patient_name,
                segment_index=seg_idx,
                predicted_stage=pred_stage,
                explanation_fig=fig,
                probabilities=probs[seg_idx],
                flag_types=flag_types,
                user_feedback=None
            )


        textual_btn.click(
            fn=run_textual_explanation,
            inputs=[select_patient_dropdown, slider, feat_dd, prob_state, flag_taxonomy],
            outputs=[textual_output],
            queue=True
        )

        # Challenge callback: re-generate taking user feedback into account
        def on_challenge(patient_name, seg_idx, sensor, probs, flag_types, previous_llm_text, user_text):

            segments = preprocess_file(patients_data[patient_name]['gait_file'], SEGMENT_LENGTH)

            fig = plot_explanation(seg_idx, segments[seg_idx], sensor)

            pred_stage = CLASS_NAMES[np.argmax(probs[seg_idx])]

            return gpt_flag_justify(
                patient_name=patient_name,
                segment_index=seg_idx,
                predicted_stage=pred_stage,
                explanation_fig=fig,
                probabilities=probs[seg_idx],
                flag_types=flag_types,
                previous_generated_justification=previous_llm_text,
                user_feedback=user_text
            )


        feedback_btn.click(
            fn=on_challenge,
            inputs=[select_patient_dropdown, slider, feat_dd, prob_state, flag_taxonomy, textual_output, feed_back_output],
            outputs=[textual_output],
        )

    demo.launch()


if __name__ == "__main__":
    main()

