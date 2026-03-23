import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load stacked data (16 bands)
# ----------------------------
with rasterio.open(
    r"E:\Projects for Resume\terrain_analyzer\src\data\processed\urban_encroachment_stack.tif"
) as src:
    data = src.read()

C, H, W = data.shape
assert C == 16, f"Expected 16 bands, got {C}"

# ----------------------------
# Correct band indices (NEW)
# ----------------------------
# After image RGB
b2_a = data[7]
b3_a = data[8]
b4_a = data[9]

# NDVI difference
ndvi_diff = data[14]

# ----------------------------
# NDVI-only baseline (binary)
# ----------------------------
ndvi_mask = (ndvi_diff < -0.05).astype(np.uint8)

# ----------------------------
# Load CNN prediction
# ----------------------------
cnn_pred = np.load(
    r"E:\Projects for Resume\terrain_analyzer\src\results\cnn_pred_map.npy"
)

# Normalize CNN prediction
cnn_norm = (cnn_pred - cnn_pred.min()) / (cnn_pred.max() - cnn_pred.min() + 1e-6)

# High-confidence CNN decision
cnn_bin = (cnn_norm > 0.85).astype(np.uint8)

# ----------------------------
# Normalize RGB for display
# ----------------------------
rgb = np.stack([b4_a, b3_a, b2_a], axis=-1)
rgb = (rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) + 1e-6)
rgb = np.clip(rgb, 0, 1)

# ----------------------------
# Side-by-side comparison
# ----------------------------
plt.figure(figsize=(18, 6))

# NDVI baseline
plt.subplot(1, 2, 1)
plt.title("NDVI-only Change Detection")
plt.imshow(rgb)
plt.imshow(ndvi_mask, cmap="Reds", alpha=0.5)
plt.axis("off")

# CNN result (FAIR comparison)
plt.subplot(1, 2, 2)
plt.title("CNN (High-confidence Prediction)")
plt.imshow(rgb)
plt.imshow(cnn_bin, cmap="Reds", alpha=0.5)
plt.axis("off")

plt.tight_layout()
plt.show()
