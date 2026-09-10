"""
Shared image-input helper for the Streamlit app (app.py).

Both the file-uploader and the camera (`st.camera_input`) hand back a
Streamlit `UploadedFile`-like object. This module normalizes either one
into a single validated PIL.Image so the rest of the app (preprocessing,
model inference, Grad-CAM, ViT attention) never needs to know which input
method produced the image — it always sees the same `PIL.Image` type.

Nothing here touches the model, the training pipeline, or the
explainability implementations; it's purely an input-handling utility.
"""
from typing import Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError


def load_image_from_upload(uploaded_file) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Convert a Streamlit-uploaded or camera-captured file into a clean,
    RGB, EXIF-corrected PIL.Image.

    Returns:
        (image, error_message) — exactly one of the two is None.
        On success: (PIL.Image, None)
        On failure: (None, "human-readable error message")
    """
    if uploaded_file is None:
        return None, "No image selected."

    try:
        image = Image.open(uploaded_file)
        # Camera photos (and many phone/DSLR uploads) embed an EXIF
        # orientation tag; without correcting for it the image can appear
        # sideways/upside-down to both the model and the Grad-CAM overlay.
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except UnidentifiedImageError:
        return None, "Unable to process this image. Please capture or upload a valid image."
    except Exception as e:  # noqa: BLE001 - surfaced to the user, not swallowed silently
        return None, f"Unable to process this image ({e})."

    if image.width < 10 or image.height < 10:
        return None, "Unable to process this image. Please capture or upload a valid image."

    return image, None
