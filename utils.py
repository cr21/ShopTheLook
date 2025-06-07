import os
i=0
def rename_images_in_directory(base_directory):
    i=0
    for root, dirs, files in os.walk(base_directory):
        
        for file in files:
            i+=1
            if file.endswith('.jpg'):
                parent_dir = os.path.basename(root)
                new_file_name = f"{parent_dir}_{file}"
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_file_name)
                # print(old_file_path)
                # print(new_file_path)
                # print(new_file_name)
                #break
                os.rename(old_file_path, new_file_path)
                # print(f"Renamed: {old_file_path} to {new_file_path}")
        print(i)
rename_images_in_directory('/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img/img_highres')
print(i)