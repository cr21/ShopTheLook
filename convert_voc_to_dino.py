import os
from multiprocessing import Pool, cpu_count
from PIL import Image



# Convert Pascal to DINO (normalized x1, y1, x2, y2)
def convert_dino_format(x1, y1, x2, y2, img_w, img_h):
    return x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h

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

    # DINO
    x1_n, y1_n, x2_n, y2_n = convert_dino_format(x1, y1, x2, y2, img_w, img_h)
    dino_line = f"{' '.join(metadata)} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f}"

    return dino_line

# Main function
def convert_pascal_to_dino_text(
    voc_path, image_root, dino_output_path
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
    dino_lines = [res for res in results if res]

    

    # Write DINO file
    with open(dino_output_path, "w") as f_dino:
        f_dino.write(f"{len(dino_lines)}\n")
        f_dino.write("image_name clothes_type pose_type x_min y_min x_max y_max\n")
        f_dino.write("\n".join(dino_lines))

    print(f"Conversion done: {len(dino_lines)} entries written.")


if __name__ == "__main__":
    # high resolution image dino format
    convert_pascal_to_dino_text(
    voc_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_high_res_inshop_1.txt",
    image_root="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark",
    dino_output_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_highres_bbox_dino_format.txt"
    )
    # low resolution image dino format
    convert_pascal_to_dino_text(
    voc_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_inshop.txt",
    image_root="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark",
    dino_output_path="/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/In-shop Clothes Retrieval Benchmark/Anno/list_bbox_dino_format.txt"
    )