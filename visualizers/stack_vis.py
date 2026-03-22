import rasterio
import matplotlib.pyplot as plt
import numpy as np

stack_path = r"E:\Projects for Resume\terrain_analyzer\src\data\processed\urban_encroachment_stack.tif"

with rasterio.open(stack_path) as src:
    data = src.read()
    print("Bands:", src.count)
    print("Shape:", data.shape)
    print("Dtype:", src.dtypes)
    print("CRS:", src.crs)
    print("Resolution:", src.res)

# Band layout:
# 0–3  : before B2,B3,B4,B8
# 4    : NDVI_before
# 5–8  : after B2,B3,B4,B8
# 9    : NDVI_after
# 10   : NDVI_diff

b2_b, b3_b, b4_b, b8_b = data[0:4]
ndvi_b = data[4]
b2_a, b3_a, b4_a, b8_a = data[5:9]
ndvi_a = data[9]
ndvi_d = data[10]

before_rgb = np.stack([b4_b, b3_b, b2_b], axis=-1)
after_rgb  = np.stack([b4_a, b3_a, b2_a], axis=-1)

def normalize(img):
    img = img - np.percentile(img, 2)
    img = img / (np.percentile(img, 98) + 1e-6)
    return np.clip(img, 0, 1)

before_rgb = normalize(before_rgb)
after_rgb  = normalize(after_rgb)

plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
plt.title("Before (2019)")
plt.imshow(before_rgb)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("After (2024)")
plt.imshow(after_rgb)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("NDVI Difference")
plt.imshow(ndvi_d, cmap="RdYlGn", vmin=-0.4, vmax=0.4)
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.show()