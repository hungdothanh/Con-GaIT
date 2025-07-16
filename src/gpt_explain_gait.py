
# ----------------- gpt_exp_gait.py-----------------
import requests
from config import OPENAI_API_KEY, CLASS_NAMES
from src.utils import convert_matplotlib_fig_to_base64


def gpt_flag_justify(patient_name,
                     segment_index,
                     predicted_stage,
                     explanation_fig,
                     probabilities,
                     flag_types,
                     previous_generated_justification=None,
                     user_feedback=None):
    # Encode explanation figures
    base64_xais = convert_matplotlib_fig_to_base64(explanation_fig)
    class_idx = CLASS_NAMES.index(predicted_stage)
    confidence = probabilities[class_idx]

    # Build the prompt
    flag_list = ', '.join(flag_types) if flag_types else 'No specific flag'
    previous_justification_section = previous_generated_justification if previous_generated_justification else ''
    feedback_section = user_feedback if user_feedback else ''
    prompt = (
        f"-- System --"
        f"You are an Explainable AI expert in Parkinson’s Disease gait‐analysis models." +

        f"-- Context --"
        f"Patient={patient_name}, Segment={segment_index}, " +
        f"Predicted Hoehn and Yahr Stage={predicted_stage} ({confidence:.1%} confidence). \nn" +

        f"Flag Types: {flag_list}, " +
        f"brief description of flag types: Factual Error (e.g., incorrect input), Normative Conflict (e.g., clinical context mismatch), or Reasoning Flaw (e.g., implausible attribution). \nn" +
        f"User feedback on your previous generated justification: {feedback_section}."
        f"Your previous generated justification {previous_justification_section} (in case user feedback is provided -> Update your points to address clinician’s concerns.)" +

        f"The attached image is explanation overlay for the selected sensor of top-5 with highlighted regions on the raw signal plot." +
        
        f"-- Task --"
        f"Your final answer should be correct, intuitive, compact, and simple for end-users to understand. " +
        f"Your final answer should be separated by bullet points (keep bullets under max 2 sentences each)." +
        
        f"Based on the attached LRP explanation image, provide justifications evaluating whether these highlighted gait patterns support or contradict the model's predicted Hoehn & Yahr stage. "
        f"Address each flagged issue in turn and offer a clear conclusion."
    )

    # Prepare image attachments
    images_payload = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        for img_b64 in base64_xais
    ]

    messages = [
        {"role": "system", "content": "You are an Explainable AI expert in Parkinson’s Disease gait analysis."},
        {"role": "user", "content": prompt, 
         "attachments": images_payload}
    ]
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 5000
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

