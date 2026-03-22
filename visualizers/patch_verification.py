import numpy as np
import matplotlib.pyplot as plt

img = np.load("dataset/images/img_0.npy")
mask = np.load("dataset/masks/mask_0.npy")

rgb = np.stack([img[2], img[1], img[0]], axis=-1)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

plt.subplot(1,2,1)
plt.imshow(rgb)
plt.title("Image Patch")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Encroachment Mask")
plt.axis("off")
plt.show()
