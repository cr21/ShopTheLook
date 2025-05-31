# In bbox location, "x_1" and "y_1" represent the upper left point coordinate of bounding box, "x_2" and "y_2" represent the lower right point coordinate of bounding box. Bounding box locations are listed in the order of [x_1, y_1, x_2, y_2].
# The bounding box format you described — [x_1, y_1, x_2, y_2] where:

# x_1, y_1 = top-left (upper-left) corner of the bounding box

# x_2, y_2 = bottom-right (lower-right) corner of the bounding box
# COCO Format: [x, y, width, height] — top-left corner + width/height

# YOLO Format: [x_center, y_center, width, height] — all values normalized (relative to image size)

# Pascal VOC Format (yours): [x_min, y_min, x_max, y_max] — absolute coordinates
from PIL import Image, ImageDraw
import os
from glob import glob

def get_bbox_on_high_res(low_res_path, low_bbox, high_res_path):
    """
    Generate a bounding box from a low-resolution image scaled onto a high-resolution image.

    Parameters:
    - low_res_path: Path to the low-resolution image (string)
    - low_bbox: Bounding box [x1, y1, x2, y2] from the low-resolution image (list of 4 integers)
    - high_res_path: Path to the high-resolution image (string)
    - output_path: Path to save the output high-resolution image with the drawn bounding box (string)

    Returns:
    - Bounding box [x1, y1, x2, y2] on the high-resolution image (list of 4 integers)
    """
    # Open the images
    low_res_image = Image.open(low_res_path)
    high_res_image = Image.open(high_res_path)

    # Get dimensions of both images
    low_w, low_h = low_res_image.size
    high_w, high_h = high_res_image.size

    # Extract low-resolution bounding box
    x1_low, y1_low, x2_low, y2_low = low_bbox

    # Scale coordinates
    x1_high = int((x1_low / low_w) * high_w)
    y1_high = int((y1_low / low_h) * high_h)
    x2_high = int((x2_low / low_w) * high_w)
    y2_high = int((y2_low / low_h) * high_h)

    return [x1_high, y1_high, x2_high, y2_high]


if __name__ == "__main__":
    low_res_path = r"/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/img"
    high_res_path = r"/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/img_highres"
    low_res_bbox_path = r"/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_inshop.txt"
    high_res_bbox_path = r"/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_high_res_inshop.txt"

    # Get the list of low-resolution images
    low_res_images = [f for f in glob(os.path.join(low_res_path,'*/*/*/*.jpg'))]

    # Get the list of high-resolution images
    high_res_images = [f for f in glob(os.path.join(high_res_path,'*/*/*/*.jpg'))]
    print(low_res_images[1:10])
    print(high_res_images[1:10])
    low_res_bbox = []
    with open(low_res_bbox_path, 'r') as f:
        lines=f.readlines()[2:]
    low_res_bbox = [[int(i) for i in line.split()[3:]] for line in lines]
    print(low_res_bbox[1:10])
    print(low_res_images[0])
    print(low_res_bbox[0])
    print(high_res_images[0])
    print(get_bbox_on_high_res(low_res_images[0], low_res_bbox[0], high_res_images[0]))