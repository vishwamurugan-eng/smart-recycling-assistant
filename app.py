"""
Streamlit Web Application for Smart Recycling Assistant.
Demonstrates TensorFlow waste image classification, scikit-learn evaluation metrics,
and live MySQL database logging & inspection.
"""

import json
import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

from db_manager import (
    DB_CONFIG,
    init_database,
    log_prediction,
    fetch_prediction_history,
    fetch_catalog_summary,
    sync_dataset_to_mysql,
)

st.set_page_config(
    page_title="Smart Recycling Assistant",
    page_icon="♻️",
    layout="wide",
)

MODEL_PATH = os.path.join("artifacts", "recycling_cnn_model.keras")
CLASSES_PATH = os.path.join("artifacts", "class_names.json")
METRICS_PATH = os.path.join("artifacts", "metrics.json")
CM_PLOT_PATH = os.path.join("artifacts", "confusion_matrix.png")
CURVES_PLOT_PATH = os.path.join("artifacts", "training_curves.png")
IMG_SIZE = (160, 160)

RECYCLING_GUIDELINES = {
    "plastic": {
        "bin": "Blue Bin — Dry Recyclable Plastic",
        "tips": "Rinse out food residue, crush bottles to save space, and keep caps attached.",
        "color": "#1E88E5",
    },
    "paper": {
        "bin": "Blue/White Bin — Paper & Cardboard",
        "tips": "Keep dry and flat. Remove plastic tape or greasy food liners before recycling.",
        "color": "#8D6E63",
    },
    "glass": {
        "bin": "Green Bin — Glass Bottles & Jars",
        "tips": "Rinse thoroughly and remove metal lids (recycle metal lids separately).",
        "color": "#2E7D32",
    },
    "metal": {
        "bin": "Grey/Yellow Bin — Metal & Aluminum Cans",
        "tips": "Empty liquids, rinse cans, and tuck sharp tin lids inside before disposal.",
        "color": "#546E7A",
    },
}


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None, ["glass", "metal", "paper", "plastic"]
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            class_names = json.load(f)
    else:
        class_names = ["glass", "metal", "paper", "plastic"]
    return model, class_names


# Sidebar: MySQL Connection Configuration
st.sidebar.header("🗄️ MySQL Database Settings")
db_host = st.sidebar.text_input("Host", value=DB_CONFIG["host"])
db_port = st.sidebar.number_input("Port", value=DB_CONFIG["port"], step=1)
db_user = st.sidebar.text_input("User", value=DB_CONFIG["user"])
db_pass = st.sidebar.text_input("Password", value=DB_CONFIG["password"], type="password")
db_name = st.sidebar.text_input("Database Name", value=DB_CONFIG["database"])

active_db_config = {
    "host": db_host,
    "port": int(db_port),
    "user": db_user,
    "password": db_pass,
    "database": db_name,
}

if st.sidebar.button("Test & Initialize MySQL Tables"):
    ok, msg = init_database(active_db_config)
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

st.title("♻️ Smart Recycling Assistant")
st.caption(
    "Supervised Waste Image Classification (TensorFlow + scikit-learn) integrated with MySQL"
)

tab_classify, tab_metrics, tab_db = st.tabs(
    [
        "📸 Classify Waste Image",
        "📊 Model Performance (scikit-learn)",
        "🗄️ MySQL Database Log",
    ]
)

with tab_classify:
    model, class_names = load_trained_model()
    if model is None:
        st.warning(
            "Trained model not found in `artifacts/`. Run `python train_classifier.py --generate-sample-data` first!"
        )
    else:
        col_upload, col_result = st.columns([1, 1.2])
        with col_upload:
            st.subheader("1. Upload Waste Item Image")
            uploaded_file = st.file_uploader(
                "Upload a photo of plastic, paper, glass, or metal waste",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        with col_result:
            st.subheader("2. Classification & Disposal Guide")
            if uploaded_file is not None:
                # Preprocess image for TensorFlow model
                img_resized = image.resize(IMG_SIZE)
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_batch = np.expand_dims(img_array, axis=0)

                probs = model.predict(img_batch, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                pred_class = class_names[top_idx]
                confidence = float(probs[top_idx]) * 100.0

                guide = RECYCLING_GUIDELINES.get(
                    pred_class.lower(),
                    {"bin": "General Recycling Bin", "tips": "Ensure item is clean and dry."},
                )

                st.metric(
                    label="Predicted Material Category",
                    value=pred_class.upper(),
                    delta=f"{confidence:.2f}% Confidence",
                )
                st.info(f"**Recommended Bin:** {guide['bin']}\n\n**Preparation Tip:** {guide['tips']}")

                # Probability distribution across all 4 classes
                prob_df = pd.DataFrame(
                    {
                        "Material": [c.capitalize() for c in class_names],
                        "Confidence (%)": np.round(probs * 100, 2),
                    }
                ).set_index("Material")
                st.bar_chart(prob_df)

                # Log prediction into MySQL
                ok, log_msg = log_prediction(
                    uploaded_file.name,
                    pred_class,
                    confidence,
                    guide["bin"],
                    config=active_db_config,
                )
                if ok:
                    st.success(f"✅ {log_msg}")
                else:
                    st.warning(f"⚠️ {log_msg}")
            else:
                st.write("Upload an image on the left to classify it and log the result to MySQL.")

with tab_metrics:
    st.subheader("Supervised Learning Evaluation (`scikit-learn` + `TensorFlow`)")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
        st.metric("Validation Accuracy", f"{metrics_data['accuracy'] * 100:.2f}%")

        report_df = pd.DataFrame(metrics_data["report"]).transpose()
        st.write("#### Per-Class Precision, Recall & F1-Score (`sklearn.metrics.classification_report`)")
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("Run `python train_classifier.py` to generate evaluation metrics.")

    col_cm, col_curves = st.columns(2)
    with col_cm:
        if os.path.exists(CM_PLOT_PATH):
            st.image(CM_PLOT_PATH, caption="scikit-learn Confusion Matrix", use_container_width=True)
    with col_curves:
        if os.path.exists(CURVES_PLOT_PATH):
            st.image(CURVES_PLOT_PATH, caption="TensorFlow Training Curves", use_container_width=True)

with tab_db:
    st.subheader("MySQL Live Database Explorer")
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("🔄 Refresh MySQL Tables"):
            st.rerun()
    with col_act2:
        if st.button("📥 Sync Local `dataset/` Images to MySQL Catalog"):
            ok, sync_msg = sync_dataset_to_mysql("dataset", config=active_db_config)
            if ok:
                st.success(sync_msg)
            else:
                st.error(sync_msg)

    st.write("#### 1. `prediction_history` Table (Recent Classifications)")
    history_df = fetch_prediction_history(limit=50, config=active_db_config)
    st.dataframe(history_df, use_container_width=True)

    st.write("#### 2. `waste_images_catalog` Table Summary (Registered Dataset Images)")
    catalog_df = fetch_catalog_summary(config=active_db_config)
    st.dataframe(catalog_df, use_container_width=True)