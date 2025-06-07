from PIL import Image
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import tqdm

def get_bbox_on_high_res(args):
    """
    Get the bounding box on the high-resolution image
    Args:
        args: tuple of (low_res_path, metadata_line, high_res_path)
    Returns:
        final_metadata: list of (image_name, clothes_type, pose_type, x_1, y_1, x_2, y_2) for high-resolution image
    """
    low_res_path, metadata_line, high_res_path = args
    try:
        # Open images
        low_res_image = Image.open(low_res_path)
        high_res_image = Image.open(high_res_path)

        # Get dimensions
        low_w, low_h = low_res_image.size
        high_w, high_h = high_res_image.size

        # Extract metadata
        parts = metadata_line.strip().split()
        metadata = parts[:3]
        metadata[0]= metadata[0].replace('img','img_highres')
        bbox = list(map(int, parts[3:]))

        # Scale bbox
        x1_high = int((bbox[0] / low_w) * high_w)
        y1_high = int((bbox[1] / low_h) * high_h)
        x2_high = int((bbox[2] / low_w) * high_w)
        y2_high = int((bbox[3] / low_h) * high_h)

        # Final metadata
        final_metadata = metadata + [str(x1_high), str(y1_high), str(x2_high), str(y2_high)]
        return ' '.join(final_metadata)
    except Exception as e:
        print(f"Error processing {low_res_path} and {high_res_path}: {e}")
        return None

if __name__ == "__main__":
    # Paths
    low_res_path = r'/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img/img'
    high_res_path = r'/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img/img_highres'
    low_res_bbox_path = r"/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/Consumer-to-shop Clothes Retrieval Benchmark/list_bbox_consumer2shop.txt"
    output_path = r"/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/Consumer-to-shop Clothes Retrieval Benchmark/list_bbox_consumer2shop_highres.txt"

    # Load low-res and high-res image paths
    low_res_images = sorted(glob(os.path.join(low_res_path, '*/*/*/*.jpg')))
    high_res_images = sorted(glob(os.path.join(high_res_path, '*/*/*/*.jpg')))

    # Load bbox metadata
    with open(low_res_bbox_path, 'r') as f:
        lines = f.readlines()[2:]
    metadata_lines = lines  # Adjust if you want fewer/more
    print(len(metadata_lines))
    print(len(low_res_images))
    print(len(high_res_images))
    # Verify consistency
    # assert len(metadata_lines) == len(low_res_images) == len(high_res_images), "Mismatch in counts"

    # Prepare args for parallel execution
    task_args = list(zip(low_res_images, metadata_lines, high_res_images))

    # Parallel processing
    print("Processing bounding boxes in parallel...")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(get_bbox_on_high_res, task_args), total=len(task_args)))

    # Filter out any failed items
    results = [line for line in results if line is not None]

    # Write output
    print(f"Writing {len(results)} entries to {output_path}")
    # output_path = output_path.replace('.txt', '_1.txt')
    with open(output_path, 'w') as f:
        f.write(f"{len(results)}\n")
        f.write("image_name clothes_type pose_type x_1 y_1 x_2 y_2\n")
        f.write('\n'.join(results))
