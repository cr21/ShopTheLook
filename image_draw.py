# In bbox location, "x_1" and "y_1" represent the upper left point coordinate of bounding box, "x_2" and "y_2" represent the lower right point coordinate of bounding box. Bounding box locations are listed in the order of [x_1, y_1, x_2, y_2].
# The bounding box format you described — [x_1, y_1, x_2, y_2] where:

# x_1, y_1 = top-left (upper-left) corner of the bounding box

# x_2, y_2 = bottom-right (lower-right) corner of the bounding box
# COCO Format: [x, y, width, height] — top-left corner + width/height

# YOLO Format: [x_center, y_center, width, height] — all values normalized (relative to image size)

# Pascal VOC Format (yours): [x_min, y_min, x_max, y_max] — absolute coordinates
from PIL import Image, ImageDraw

def draw_scaled_bbox_on_high_res(low_res_path, low_bbox, high_res_path, output_path):
    """
    Draws a bounding box from a low-resolution image scaled onto a high-resolution image.

    Parameters:
    - low_res_path: Path to the low-resolution image (string)
    - low_bbox: Bounding box [x1, y1, x2, y2] from the low-resolution image (list of 4 integers)
    - high_res_path: Path to the high-resolution image (string)
    - output_path: Path to save the output high-resolution image with the drawn bounding box (string)
    
    Returns:
    - output_path: The path where the modified high-resolution image is saved
    """
    # Open the images
    low_res_image = Image.open(low_res_path)
    high_res_image = Image.open(high_res_path)

    # Convert to RGB if the image has transparency (RGBA)
    if high_res_image.mode == 'RGBA':
        high_res_image = high_res_image.convert('RGB')

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

    # Draw bounding box on the high-resolution image
    draw = ImageDraw.Draw(high_res_image)
    draw.rectangle([x1_high, y1_high, x2_high, y2_high], outline="blue", width=2)
    # draw.rectangle([0, 50, 331, 624], outline="red", width=2)
    # draw.text((x1_high, y1_high), "Hello", fill="red")

    # Save the output image
    high_res_image.save(output_path)

    return output_path,[x1_high, y1_high, x2_high, y2_high]


if __name__ == "__main__":

    #high_res_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/img_highres/WOMEN/Cardigans/id_00000070/03_4_full.jpg"
    #low_res_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/img/WOMEN/Cardigans/id_00000070/03_4_full.jpg"
    low_res_path =  '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/img/WOMEN/Dresses/id_00000035/04_2_side.jpg'


    high_res_path =  '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/img_highres/WOMEN/Dresses/id_00000035/id_00000035_04_2_side.jpg'

    # low_bbox =[285, 95, 427, 128]
    low_bbox =[100 ,39 ,156 ,256]
    #[1], [235, 251, 612, 904]           
    ## [175, 525, 356, 702], [109, 388, 220, 489], [224, 203, 280, 239], [198, 258, 399, 507]
    # /home/ubuntu/data/images/WOMEN_Blouses_Shirts_id_00002532_04_4_full.jpg 1 0.442667 0.653043 0.285333 0.406903
    # /home/ubuntu/data/images/WOMEN_Blouses_Shirts_id_00002532_04_4_full.jpg 3 0.461333 0.925976 0.202667 0.129882

    #high_res_path =
    output_path = r"/Users/chiragtagadiya/Documents/y_out_1.jpg"

    print(draw_scaled_bbox_on_high_res(low_res_path, low_bbox, high_res_path, output_path))

    ## OLD CODE

    from PIL import Image, ImageDraw

    # Load the image
    image_path = low_res_path
    #image_path ='/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Category and Attribute/img/Sweet_Crochet_Blouse/img_00000070.jpg'
    image = Image.open(image_path)

    # Convert to RGB if the image has transparency (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Define the bounding box using the coordinates (left, top, right, bottom)
    bounding_box = low_bbox
    # Draw the bounding box on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle(bounding_box, outline="green", width=3)

    # Display the modified image
    #image.show()
    image.save(r"/Users/chiragtagadiya/Documents/y_out_1_1.jpg")
