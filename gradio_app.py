"""
Gradio Web Application for Skin Cancer Detection (CNN-SVM).
Provides an interactive web interface with drag-and-drop image upload,
OpenCV CLAHE preprocessing preview, and multi-class probability breakdown.
"""

import os
import json
import numpy as np
import gradio as gr
from PIL import Image

from src.predict import predict_skin_lesion
from src.preprocessing import CLASS_NAMES

def classify_skin_lesion(input_image, apply_clahe: bool = True):
    if input_image is None:
        return None, {}, "Please upload an image."

    res = predict_skin_lesion(input_image)
    if res.get("status") != "success":
        return None, {}, f"Error: {res.get('error_message')}"

    # Probabilities dict for Gradio Label component (formatted as 0.0 to 1.0)
    label_probs = {k: v / 100.0 for k, v in res["probabilities"].items()}

    summary_text = f"""### Predicted: **{res['predicted_name']}** ({res['predicted_code'].upper()})
- **Severity**: {res['severity']}
- **Confidence**: {res['confidence']:.1f}%
- **Clinical Risk Level**: {res['risk_level']}
- **Recommendation**: {res['recommendation']}

*Disclaimer: {res['disclaimer']}*
"""
    return res["preprocessed_image"], label_probs, summary_text


# Build Gradio Blocks Interface
with gr.Blocks(title="DermoScan AI - Skin Cancer Detection (CNN-SVM)") as demo:
    gr.Markdown("# 🔬 DermoScan AI: Skin Cancer Detection System")
    gr.Markdown("Automated dermoscopic classification using Deep CNN Feature Extraction (MobileNetV2) and a Support Vector Machine (SVM) classifier.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Skin Lesion Image")
            clahe_checkbox = gr.Checkbox(value=True, label="Apply OpenCV CLAHE Contrast Enhancement")
            submit_btn = gr.Button("Analyze Lesion", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Preprocessed Image (128x128)")
            output_labels = gr.Label(num_top_classes=5, label="Top Predictions")
            output_markdown = gr.Markdown()

    submit_btn.click(
        fn=classify_skin_lesion,
        inputs=[image_input, clahe_checkbox],
        outputs=[output_image, output_labels, output_markdown]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
