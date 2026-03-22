import rasterio
import matplotlib.pyplot as plt
import numpy as np
with rasterio.open(r"E:\Projects for Resume\terrain_analyzer\src\data\processed\encroachment_labels.tif") as src:
    labels = src.read(1)

plt.imshow(labels, cmap="gray")
plt.title("Encroachment Candidate Mask")
plt.axis("off")
plt.show()
