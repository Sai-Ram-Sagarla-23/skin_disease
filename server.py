#!/usr/bin/env python
"""
Inference server for index.html.

Reuses exactly the same model, preprocessing and explainability code as app.py —
nothing about the pipeline changes, it is just exposed over HTTP so the browser
front-end can call it.

    pip install fastapi uvicorn python-multipart
    python server.py

Then open index.html and set the server address to http://localhost:8000
"""
import base64
import io

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image

from src.dataset import CLASS_FULL_NAMES, CLASS_NAMES
from src.explainability.attention_visualization import get_cls_attention_map, overlay_attention
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.hybrid_model import HybridResNetViT
from src.preprocessing import get_val_test_transform
from src.utils.image_input import load_image_from_upload

app = FastAPI(title="Skin Lesion Classifier")

# The page is opened from a local file or a local static server, so origins vary.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {}


def _to_data_uri(array_like) -> str:
    """Accepts a float [0,1] or uint8 HxWx3 array and returns a PNG data URI."""
    arr = np.asarray(array_like)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.on_event("startup")
def load_everything():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = hf_hub_download(repo_id="sai2318/best", filename="best_model.pth")

    m_cfg = cfg["model"]
    model = HybridResNetViT(
        num_classes=m_cfg["num_classes"],
        resnet_backbone=m_cfg["resnet_backbone"],
        vit_backbone=m_cfg["vit_backbone"],
        fusion_dim=m_cfg["fusion_dimension"],
        fusion_type="attention",
        dropout=m_cfg["dropout"],
        pretrained=False,
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    STATE.update(
        cfg=cfg,
        model=model,
        transform=get_val_test_transform(cfg),
        device=device,
    )
    print(f"Model ready on {device}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(STATE.get("device"))}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not STATE:
        raise HTTPException(503, "Model is still loading")

    raw = await file.read()
    image, err = load_image_from_upload(io.BytesIO(raw))
    if err:
        raise HTTPException(400, err)

    cfg, model = STATE["cfg"], STATE["model"]
    device, transform = STATE["device"], STATE["transform"]

    tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.resnet_branch.layer4)
    cam, pred_idx = gradcam.generate(tensor)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    pred_class = CLASS_NAMES[pred_idx]
    size = cfg["data"]["image_size"]
    original = np.array(image.resize((size, size))) / 255.0

    attn_weights = model.vit_branch.get_last_attention()
    attn_map = get_cls_attention_map(attn_weights, size) if attn_weights is not None else None
    attn_overlay = overlay_attention(original, attn_map) if attn_map is not None else original

    return {
        "pred_class": pred_class,
        "pred_name": CLASS_FULL_NAMES[pred_class],
        "confidence": float(probs[pred_idx]),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "original": _to_data_uri(original),
        "gradcam": _to_data_uri(overlay_heatmap(original, cam)),
        "attention": _to_data_uri(attn_overlay),
    }


if __name__ == "__main__":
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
