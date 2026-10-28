"""
Streamlit Web Application: Skin Cancer Detection and Classification System (CNN-SVM)
Author: Educational Research Prototype
Description: Web interface allowing users to upload or select dermoscopic skin lesion images,
preprocess them with OpenCV (CLAHE enhancement), extract CNN representations, and classify
them using a trained Support Vector Machine (SVM) classifier.
"""

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="DermoScan AI - Skin Cancer Detection (CNN-SVM)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        color: #991b1b;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-malignant {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-benign {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-precancer {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading CNN-SVM diagnostic pipeline...")
def load_cached_predictor():
    """Load predictor model instance once and cache in memory."""
    from src.predict import SkinCancerPredictor
    return SkinCancerPredictor()


def main():
    # Header Section
    st.markdown('<div class="main-header">🔬 DermoScan AI: Skin Cancer Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Dermoscopic Lesion Classification powered by Deep CNN Feature Extraction & Support Vector Machines (SVM)</div>', unsafe_allow_html=True)

    # Disclaimer Alert
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Important Medical Disclaimer:</strong> This application is an educational and research prototype developed using the HAM10000 dataset.
        It is <strong>NOT</strong> an FDA-cleared or CE-marked medical device and must never replace direct evaluation, dermoscopy, or biopsy by a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Architecture")
        st.markdown("""
        - **Feature Extractor**: Deep CNN (MobileNetV2 Backbone, ImageNet pre-trained + Fine-Tuned)
        - **Classifier**: Support Vector Machine (RBF Kernel, Balanced Class Weighting)
        - **Preprocessing**: OpenCV CLAHE (L-channel adaptive histogram equalization), Resizing to 128×128, Normalization
        - **Dataset**: HAM10000 (10,015 dermoscopic records across 7 diagnostic categories)
        """)

        # Model Performance Metrics
        metrics_file = os.path.join("reports", "evaluation_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                eval_data = json.load(f)
            svm_m = eval_data.get("svm_model", {})
            st.markdown("---")
            st.subheader("📊 Test Set Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{svm_m.get('accuracy', 0)*100:.1f}%")
            col2.metric("Weighted F1", f"{svm_m.get('f1_weighted', 0)*100:.1f}%")
            col3, col4 = st.columns(2)
            col3.metric("Balanced Acc", f"{svm_m.get('balanced_accuracy', 0)*100:.1f}%")
            col4.metric("ROC-AUC", f"{svm_m.get('roc_auc_macro', 0):.3f}")

        st.markdown("---")
        st.subheader("📚 Diagnostic Classes")
        st.markdown("""
        1. **MEL**: Melanoma *(Malignant)*
        2. **BCC**: Basal Cell Carcinoma *(Malignant)*
        3. **AKIEC**: Actinic Keratoses *(Pre-cancerous)*
        4. **NV**: Melanocytic Nevi *(Benign)*
        5. **BKL**: Benign Keratosis *(Benign)*
        6. **VASC**: Vascular Lesions *(Benign)*
        7. **DF**: Dermatofibroma *(Benign)*
        """)

    # Main Content Tabs
    tab_predict, tab_evaluation, tab_guide = st.tabs(["🔍 Lesion Analysis", "📈 Model Evaluation & Reports", "📖 ABCDE Clinical Guide"])

    with tab_predict:
        col_input, col_results = st.columns([1, 1.2])

        with col_input:
            st.subheader("1. Provide Lesion Image")
            input_mode = st.radio("Choose Input Method:", ["Upload Image File", "Select Preloaded Benchmark Sample"], horizontal=True)

            img_to_analyze = None
            img_name = ""

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Upload a skin lesion dermoscopic image (JPG, JPEG, PNG, WEBP):",
                    type=["jpg", "jpeg", "png", "webp"]
                )
                if uploaded_file is not None:
                    # Validate file size (< 25MB)
                    if uploaded_file.size > 25 * 1024 * 1024:
                        st.error("Error: Uploaded file exceeds maximum allowed size of 25MB.")
                    else:
                        try:
                            img_to_analyze = Image.open(uploaded_file)
                            img_name = uploaded_file.name
                        except Exception as e:
                            st.error(f"Invalid or corrupt image file: {e}")
            else:
                # Preloaded samples
                sample_options = {
                    "Sample Melanoma (Malignant mel)": "HAM10000_images_part_1/ISIC_0024306.jpg",
                    "Sample Basal Cell Carcinoma (Malignant bcc)": "HAM10000_images_part_1/ISIC_0024331.jpg",
                    "Sample Melanocytic Nevi (Benign mole nv)": "HAM10000_images_part_1/ISIC_0024310.jpg",
                    "Sample Benign Keratosis (Benign bkl)": "HAM10000_images_part_1/ISIC_0024312.jpg"
                }
                # Fallback to existing files
                valid_samples = {k: v for k, v in sample_options.items() if os.path.exists(v)}
                if not valid_samples:
                    # Search dynamically
                    import glob
                    found = glob.glob("HAM10000_images_part_1/*.jpg")[:4]
                    valid_samples = {f"Test Sample #{i+1}": p for i, p in enumerate(found)}

                selected_sample = st.selectbox("Select Benchmark Case:", list(valid_samples.keys()))
                if selected_sample:
                    sample_path = valid_samples[selected_sample]
                    img_to_analyze = Image.open(sample_path)
                    img_name = os.path.basename(sample_path)

            apply_clahe_option = st.checkbox("Apply CLAHE Contrast Enhancement", value=True,
                                             help="Enhances lesion border contrast in LAB color space using OpenCV.")

            if img_to_analyze is not None:
                st.image(img_to_analyze, caption=f"Selected Lesion: {img_name}", use_container_width=True)

        with col_results:
            st.subheader("2. Diagnostic Analysis & Classification")
            if img_to_analyze is not None:
                with st.spinner("Analyzing lesion via CNN feature extraction and SVM inference..."):
                    try:
                        predictor = load_cached_predictor()
                        result = predictor.predict(img_to_analyze, enhance=apply_clahe_option)
                    except Exception as e:
                        st.error(f"Inference pipeline error: {e}")
                        result = None

                if result and result.get("status") == "success":
                    pred_code = result["predicted_code"]
                    pred_name = result["predicted_name"]
                    conf = result["confidence"]
                    severity = result["severity"]
                    risk_level = result["risk_level"]
                    recommendation = result["recommendation"]

                    # Severity Badge
                    if severity == "Malignant":
                        badge_html = f'<span class="badge-malignant">⚠️ {severity.upper()}</span>'
                    elif severity.startswith("Pre-cancerous"):
                        badge_html = f'<span class="badge-precancer">⚡ {severity.upper()}</span>'
                    else:
                        badge_html = f'<span class="badge-benign">✅ {severity.upper()}</span>'

                    st.markdown(f"""
                    ### Predicted Diagnosis: **{pred_name}** (`{pred_code.upper()}`)  
                    {badge_html} &nbsp;&nbsp; <strong>Model Confidence:</strong> `{conf:.1f}%`
                    """, unsafe_allow_html=True)

                    st.info(f"**Clinical Status:** {risk_level}\n\n**Actionable Guidance:** {recommendation}")

                    # Visual Comparison: Preprocessed Image
                    st.write("**OpenCV Preprocessed Image (128×128):**")
                    col_p1, col_p2 = st.columns([1, 2])
                    with col_p1:
                        st.image(result["preprocessed_image"], width=128, caption="Resized & Enhanced")
                    with col_p2:
                        st.caption("Normalized pixel values [0, 1] scaled from CLAHE-equalized LAB color space, feeding the 1280-dimensional CNN feature extractor.")

                    # Probabilities Breakdown
                    st.write("---")
                    st.write("##### Diagnostic Probability Breakdown across 7 Classes:")
                    probs_df = pd.DataFrame([
                        {"Category": k, "Probability (%)": v} for k, v in result["probabilities"].items()
                    ])
                    st.bar_chart(probs_df.set_index("Category"))

                    # Download Report
                    json_report = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download JSON Diagnostic Report",
                        data=json_report,
                        file_name=f"diagnostic_report_{img_name}.json",
                        mime="application/json"
                    )
                elif result and result.get("status") == "error":
                    st.error(f"Classification failed: {result.get('error_message')}")
            else:
                st.info("👈 Upload an image or select a benchmark sample to run prediction.")

    with tab_evaluation:
        st.subheader("Model Evaluation, Comparison & Reports")
        st.markdown("Independent evaluation on the untouched hold-out test set (no lesion overlap / zero data leakage).")

        # Metric Comparison Table
        metrics_file = os.path.join("reports", "evaluation_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
            comp_df = pd.DataFrame({
                "Metric": ["Accuracy", "Balanced Accuracy", "Weighted F1", "Macro F1", "Weighted Recall", "Macro ROC-AUC"],
                "CNN-SVM Pipeline": [
                    f"{metrics['svm_model'].get('accuracy', 0)*100:.2f}%",
                    f"{metrics['svm_model'].get('balanced_accuracy', 0)*100:.2f}%",
                    f"{metrics['svm_model'].get('f1_weighted', 0)*100:.2f}%",
                    f"{metrics['svm_model'].get('f1_macro', 0)*100:.2f}%",
                    f"{metrics['svm_model'].get('recall_weighted', 0)*100:.2f}%",
                    f"{metrics['svm_model'].get('roc_auc_macro', 0):.3f}"
                ],
                "CNN-Only Baseline": [
                    f"{metrics['cnn_baseline'].get('accuracy', 0)*100:.2f}%",
                    f"{metrics['cnn_baseline'].get('balanced_accuracy', 0)*100:.2f}%",
                    f"{metrics['cnn_baseline'].get('f1_weighted', 0)*100:.2f}%",
                    f"{metrics['cnn_baseline'].get('f1_macro', 0)*100:.2f}%",
                    f"{metrics['cnn_baseline'].get('recall_weighted', 0)*100:.2f}%",
                    f"{metrics['cnn_baseline'].get('roc_auc_macro', 0):.3f}"
                ]
            })
            st.table(comp_df)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            cm_img_path = os.path.join("reports", "confusion_matrix.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrix (Normalized & Counts)", use_container_width=True)
        with col_g2:
            roc_img_path = os.path.join("reports", "roc_curves.png")
            if os.path.exists(roc_img_path):
                st.image(roc_img_path, caption="One-vs-Rest ROC Curves", use_container_width=True)

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            comp_img_path = os.path.join("reports", "model_comparison_bar.png")
            if os.path.exists(comp_img_path):
                st.image(comp_img_path, caption="CNN-SVM vs CNN Baseline Comparison", use_container_width=True)
        with col_g4:
            mis_img_path = os.path.join("reports", "misclassified_analysis.png")
            if os.path.exists(mis_img_path):
                st.image(mis_img_path, caption="Misclassified Cases Failure Analysis", use_container_width=True)

        rep_txt_path = os.path.join("reports", "classification_report.txt")
        if os.path.exists(rep_txt_path):
            with open(rep_txt_path) as f:
                rep_content = f.read()
            st.text_area("Full Classification Report", rep_content, height=280)

    with tab_guide:
        st.subheader("Clinical Dermatology ABCDE Melanoma Self-Check Guide")
        st.markdown("""
        Dermatologists use the **ABCDE rule** to evaluate pigmented lesions for potential malignancy:
        
        - **A - Asymmetry**: One half of the mole does not match the other half in shape or contour.
        - **B - Border**: The edges are irregular, ragged, notched, scalloped, or blurred.
        - **C - Color**: The color is not uniform; shades of brown, black, patches of pink, red, white, or blue may appear.
        - **D - Diameter**: The spot is larger than 6 millimeters across (about the size of a pencil eraser), though melanomas can sometimes be smaller.
        - **E - Evolving**: The mole is changing in size, shape, color, elevation, or presents new symptoms such as itching or bleeding.
        """)
        st.warning("If you notice any of these signs on your skin, schedule an appointment with a board-certified dermatologist promptly.")


if __name__ == "__main__":
    main()
