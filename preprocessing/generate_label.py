import rasterio
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"E:/Projects for Resume/terrain_analyzer/src")

stack_path = BASE_DIR / "data" / "processed" / "urban_encroachment_stack.tif"
out_path   = BASE_DIR / "data" / "processed" / "encroachment_labels.tif"

with rasterio.open(stack_path) as src:
    data = src.read()
    meta = src.meta

ndvi_diff = data[14]   # Band 15
ndbi_diff = data[15]

# Rule: vegetation loss threshold
labels = np.zeros_like(ndvi_diff, dtype=np.uint8)
labels[(ndvi_diff < -0.05) & (ndbi_diff > 0.1)] = 1   # encroachment candidate

meta.update(count=1, dtype="uint8")

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(labels, 1)

print("Saved label mask:", out_path)
