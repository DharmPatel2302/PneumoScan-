import streamlit as st

import torch

from PIL import Image

import matplotlib.pyplot as plt

from utils.model_loader import load_model

from utils.preprocessing import (
    transform,
    denormalize
)

from utils.gradcam import (
    GradCAM,
    create_overlay
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "model/final_model.pt"


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🫁 Pneumonia Detection using AI"
)

st.write(
    "Upload a chest X-ray to detect pneumonia "
    "and visualize the model's decision using Grad-CAM."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    return load_model(
        MODEL_PATH,
        DEVICE
    )


try:

    model = get_model()

except Exception as e:

    st.error(
        "❌ Failed to load the model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# GRAD-CAM
# ============================================================

# DenseNet121 final feature block
target_layer = model.features.denseblock4
print("Grad-CAM target layer:", target_layer)

gradcam = GradCAM(
    model,
    target_layer
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    st.subheader(
        "Uploaded X-ray"
    )

    st.image(
        image,
        width=500
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    input_tensor = transform(
        image
    )

    # Add batch dimension

    input_tensor = input_tensor.unsqueeze(
        0
    )

    # Move to GPU/CPU

    input_tensor = input_tensor.to(
        DEVICE
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            input_tensor
        )

        # Binary classification
        probability = torch.sigmoid(
            output[0, 0]
        ).item()


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if probability >= 0.5:

        prediction = "PNEUMONIA"

        confidence = probability

    else:

        prediction = "NORMAL"

        confidence = 1 - probability


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🩺 Prediction"
    )


    col1, col2 = st.columns(2)


    with col1:

        if prediction == "PNEUMONIA":

            st.error(
                f"🫁 {prediction}"
            )

        else:

            st.success(
                f"✅ {prediction}"
            )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # Probability bar

    st.write(
        f"Pneumonia probability: "
        f"**{probability * 100:.2f}%**"
    )

    st.progress(
        probability
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Explainable AI — Grad-CAM"
    )

    st.write(
        "The highlighted regions show the areas "
        "that contributed most strongly to the model's "
        "prediction."
    )


    # Generate heatmap

    heatmap = gradcam.generate(
        input_tensor
    )


    # Denormalize image

    image_np = denormalize(
        input_tensor[0]
    )


    # Create overlay

    overlay = create_overlay(
        image_np,
        heatmap
    )


    # ========================================================
    # DISPLAY XAI
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.image(
            image_np,
            caption="Original X-ray",
            use_container_width=True
        )


    with col2:

        fig, ax = plt.subplots()

        ax.imshow(
            heatmap,
            cmap="jet"
        )

        ax.axis("off")

        ax.set_title(
            "Grad-CAM Heatmap"
        )

        st.pyplot(
            fig
        )

        plt.close(fig)


    with col3:

        st.image(
            overlay,
            caption="X-ray + Grad-CAM",
            use_container_width=True
        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 Model Explanation"
    )


    if prediction == "PNEUMONIA":

        st.warning(
            "The model detected features associated "
            "with pneumonia. The Grad-CAM visualization "
            "shows the regions that contributed to this "
            "prediction."
        )

    else:

        st.info(
            "The model classified this X-ray as normal. "
            "The Grad-CAM visualization shows the regions "
            "that influenced this prediction."
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ This application is intended for research "
        "and educational purposes only. It is not a "
        "medical diagnostic tool."
    )