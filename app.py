#!/usr/bin/env python
"""
Optional local Streamlit web interface (requirement #30).
100% free, runs entirely locally — no API calls, no internet required
after the model/dataset have been downloaded once.

Supports two input methods that both feed the SAME trained model, the SAME
preprocessing pipeline, and the SAME Grad-CAM / ViT attention explainability
code:
    1. Upload Image  (st.file_uploader — original functionality, unchanged)
    2. Take Photo     (st.camera_input — new, browser-native, key-free)

Run:
    streamlit run app.py
"""
import os

import numpy as np
import streamlit as st
import torch
import yaml

from src.dataset import CLASS_FULL_NAMES, CLASS_NAMES
from src.explainability.attention_visualization import get_cls_attention_map, overlay_attention
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.hybrid_model import HybridResNetViT
from src.preprocessing import get_val_test_transform
from src.utils.image_input import load_image_from_upload

st.set_page_config(page_title="Skin Disease Classifier", layout="wide")


@st.cache_resource
def load_everything():
    """Unchanged from the original app — same config, same architecture,
    same checkpoint loading. Cached so switching input methods doesn't
    reload the model."""
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(cfg["project"]["checkpoint_dir"], "best_model.pth")

    m_cfg = cfg["model"]
    model = HybridResNetViT(
        num_classes=m_cfg["num_classes"], resnet_backbone=m_cfg["resnet_backbone"],
        vit_backbone=m_cfg["vit_backbone"], fusion_dim=m_cfg["fusion_dimension"],
        fusion_type="attention", dropout=m_cfg["dropout"], pretrained=False,
    ).to(device)

    model_ready = os.path.exists(ckpt_path)
    if model_ready:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

    transform = get_val_test_transform(cfg)
    return cfg, model, transform, device, model_ready


def get_input_image(source: str):
    """Returns (image, error_message, is_camera_source).

    Both branches funnel into the SAME `load_image_from_upload()` helper,
    so an uploaded file and a camera capture become the exact same
    validated, RGB, EXIF-corrected PIL.Image type before anything else in
    the app touches them.
    """
    if source == "📁 Upload Image":
        uploaded = st.file_uploader(
            "Upload a dermatoscopic skin lesion image", type=["jpg", "jpeg", "png"]
        )
        if uploaded is None:
            return None, None, False
        image, err = load_image_from_upload(uploaded)
        return image, err, False

    else:  # "📷 Take Photo Using Camera"
        st.caption("📷 Take a photo of the skin lesion")
        try:
            captured = st.camera_input("Camera")
        except Exception:
            # st.camera_input can raise in environments where the browser
            # denies camera access or the component fails to initialize
            # (e.g. some embedded/headless contexts).
            st.error(
                "Camera input is not available in this browser/session. "
                "Please use image upload instead."
            )
            return None, None, True

        if captured is None:
            return None, None, True

        image, err = load_image_from_upload(captured)
        return image, err, True


def run_prediction_and_explain(image, cfg, model, transform, device, is_camera_source: bool):
    """The SAME inference + Grad-CAM + ViT-attention pipeline used by the
    original upload-only app, now called for both input methods so there
    is exactly one prediction code path (requirement #3 / #4)."""
    input_tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.resnet_branch.layer4)
    cam, pred_idx = gradcam.generate(input_tensor)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    attn_weights = model.vit_branch.get_last_attention()
    attn_map = get_cls_attention_map(attn_weights, cfg["data"]["image_size"]) if attn_weights is not None else None

    st.markdown("---")
    st.subheader("Prediction Result")
    st.caption(
        "**Academic Research Prototype:** This system is intended for educational and "
        "research purposes only and should not be used as a medical diagnosis or "
        "treatment tool."
    )

    if is_camera_source:
        st.info(
            "ℹ️ Camera photographs may differ significantly from the dermoscopic images "
            "used during training (e.g. HAM10000). Camera predictions are therefore "
            "intended only for research demonstration and may not generalize to "
            "real-world diagnosis."
        )

    col_pred, col_conf = st.columns(2)
    col_pred.metric("Predicted Disease", f"{CLASS_FULL_NAMES[pred_class]} ({pred_class})")
    col_conf.metric("Confidence", f"{confidence*100:.2f}%")

    orig = np.array(image.resize((cfg["data"]["image_size"], cfg["data"]["image_size"]))) / 255.0
    cam_overlay = overlay_heatmap(orig, cam)
    attn_overlay = overlay_attention(orig, attn_map) if attn_map is not None else orig

    st.subheader("Explainability")
    col1, col2, col3 = st.columns(3)
    col1.image(orig, caption="Original Image", use_container_width=True)
    col2.image(cam_overlay, caption="ResNet50 Grad-CAM", use_container_width=True)
    col3.image(attn_overlay, caption="Vision Transformer Attention", use_container_width=True)

    st.subheader("Full class probabilities")
    st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})


def main():
    st.title("🔬 Skin Disease Classifier")
    st.caption(
        "Hybrid ResNet50 + Vision Transformer with attention-based fusion "
        "and Grad-CAM / attention explainability."
    )
    st.warning(
        "⚠️ This system is an academic research prototype and is **not** a "
        "medical diagnosis tool. Always consult a qualified dermatologist "
        "for any real skin concern."
    )

    cfg, model, transform, device, model_ready = load_everything()

    if not model_ready:
        st.error(
            "No trained model checkpoint found at "
            f"`{cfg['project']['checkpoint_dir']}/best_model.pth`. "
            "Train the model first: `python scripts/train_hybrid.py`."
        )
        return

    st.subheader("Choose Image Source")
    source = st.radio(
        label="Image source",
        options=["📁 Upload Image", "📷 Take Photo Using Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image, error, is_camera_source = get_input_image(source)

    if error is not None:
        st.error(error)
        return

    if image is None:
        # Nothing selected/captured yet — do not attempt inference, no crash.
        st.info("Upload an image or take a photo to begin.")
        return

    st.subheader("Image Preview")
    st.image(image, caption="Captured Image" if is_camera_source else "Uploaded Image", width=320)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Running inference..."):
            run_prediction_and_explain(image, cfg, model, transform, device, is_camera_source)


if __name__ == "__main__":
    main()
