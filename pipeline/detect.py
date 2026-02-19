import numpy as np
import pandas as pd
from PIL import Image
from deepforest import main as deepforest_main


_model = None


def get_model() -> deepforest_main.deepforest:
    """Lazily load the DeepForest model (downloads weights on first call)."""
    global _model
    if _model is None:
        _model = deepforest_main.deepforest()
        _model.load_model(model_name="weecology/deepforest-tree", revision="main")
    return _model


def detect_trees(
    image: Image.Image,
    score_threshold: float = 0.3,   # back to DeepForest default
    min_box_px: int = 30,           # ~4.5m canopy minimum — skip shrubs, keep real trees
    max_box_px: int = 300,          # ignore huge detections (misidentified buildings, etc.)
    nms_thresh: float = 0.3,        # raised from 0.15 to keep nearby trees instead of merging them
) -> pd.DataFrame:
    """Run tree detection on a PIL image.

    Returns a DataFrame with columns: xmin, ymin, xmax, ymax, score, label.
    """
    model = get_model()
    model.config.nms_thresh = nms_thresh
    img_array = np.array(image).astype(np.uint8)
    results = model.predict_tile(image=img_array, patch_size=600, patch_overlap=0.25)  # larger patches + more overlap for scale=2 imagery
    if results is None or results.empty:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

    # Filter by confidence score
    results = results[results["score"] >= score_threshold]

    # Filter by bounding box size in pixels — removes noise and misidentified large structures
    widths = results["xmax"] - results["xmin"]
    heights = results["ymax"] - results["ymin"]
    size_mask = (
        (widths >= min_box_px) & (widths <= max_box_px) &
        (heights >= min_box_px) & (heights <= max_box_px)
    )
    results = results[size_mask]

    return results.reset_index(drop=True)
