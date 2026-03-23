import rasterio
import numpy as np
STACK_PATH = r"E:\Projects for Resume\terrain_analyzer\src\data\processed\urban_encroachment_stack.tif"

data = [(np.float64(210.90161290322578), np.float64(216.76612903225896)), 
        (np.float64(2304.974193548387), np.float64(344.4048387096782)), 
        (np.float64(107.19516129032263), np.float64(1640.7354838709678)), 
        (np.float64(326.57419354838714), np.float64(2031.6290322580644)), 
        (np.float64(3.488709677419365), np.float64(444.1225806451621))]
def convert_pixel_to_coordinate(px, py, stack_path):
    with rasterio.open(STACK_PATH) as src:
        lon, lat = src.transform * (px, py)
        return lon, lat

for (px, py) in data:
    lon, lat = convert_pixel_to_coordinate(px, py, STACK_PATH)
    print("Latitude, longitude:", lat, lon)
    print("\n")

# MANUAL_NEGATIVES = [
#     {"x": 674,  "y": 632},
#     {"x": 2010, "y": 1058},
#     {"x": 211,  "y": 217},
#     {"x": 2216, "y": 344},   # clamped from x=2305
#     {"x": 107,  "y": 1641},
# ]

# PATCH_SIZE = 256

# with rasterio.open(STACK_PATH) as src:
#     _, H, W = src.read().shape

# print("Image size:", W, H)

# for p in MANUAL_NEGATIVES:
#     ok_x = p["x"] + PATCH_SIZE < W
#     ok_y = p["y"] + PATCH_SIZE < H
#     print(p, "OK_X:", ok_x, "OK_Y:", ok_y)