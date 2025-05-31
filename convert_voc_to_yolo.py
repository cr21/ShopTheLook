import os
from multiprocessing import Pool, cpu_count
from PIL import Image

# Convert Pascal to YOLO (normalized x_center, y_center, width, height)
def convert_yolo_format(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height



# Worker function
def process_line(line_data):
    image_root, line = line_data
    parts = line.strip().split()
    img_path = os.path.join(image_root, parts[0])
    metadata = parts[:3]
    x1, y1, x2, y2 = map(int, parts[3:7])

    try:
        img_w, img_h = Image.open(img_path).size
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

    # YOLO
    x_c, y_c, w, h = convert_yolo_format(x1, y1, x2, y2, img_w, img_h)
    yolo_line = f"{' '.join(metadata)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

    return yolo_line

# Main function
def convert_pascal_to_yolo_text(
    voc_path, image_root, yolo_output_path
):
    with open(voc_path, "r") as f:
        lines = f.readlines()

    header = lines[1].strip()
    data_lines = lines[2:]
    num_images = lines[0].strip()
    task_data = [(image_root, line) for line in data_lines]
    print(cpu_count())
    with Pool(cpu_count()) as pool:
        results = pool.map(process_line, task_data)

    # Filter valid results
    yolo_lines = [res for res in results if res]

    # Write YOLO file
    with open(yolo_output_path, "w") as f_yolo:
        f_yolo.write(f"{len(yolo_lines)}\n")
        f_yolo.write("image_name clothes_type pose_type x_center y_center width height\n")
        f_yolo.write("\n".join(yolo_lines))


    print(f"Conversion done: {len(yolo_lines)} entries written.")


if __name__ == "__main__":
    # high resolution image yolo format
    convert_pascal_to_yolo_text(
    voc_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_high_res_inshop_1.txt",
    image_root="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark",
    yolo_output_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_highres_bbox_yolo_format.txt"
    )
    # low resolution image yolo format
    convert_pascal_to_yolo_text(
    voc_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_inshop.txt",
    image_root="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark",
    yolo_output_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_yolo_format.txt"
    )