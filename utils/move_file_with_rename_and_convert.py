import os
import shutil
from PIL import Image
from pillow_heif import open_heif

def convert_heic_to_jpg(heic_path, jpg_path):
    heif_image = open_heif(heic_path)
    img = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
    img.save(jpg_path, "JPEG")

def movefile_with_rename_and_convert(source_dir, dest_dir, new_name_pattern):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    files = os.listdir(source_dir)
    
    for index, filename in enumerate(files):
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.heic':  # Only process HEIC files
            heic_path = os.path.join(source_dir, filename)
            jpg_filename = f"{new_name_pattern}_{index+1}.jpg"
            jpg_path = os.path.join(dest_dir, jpg_filename)
            
            convert_heic_to_jpg(heic_path, jpg_path)
            
            print(f"Converted {filename} to {jpg_filename}")
        else:
            print(f"Skipping non-HEIC file: {filename}")


if __name__ == "__main__":
    class_list = ["montfleur", "minere", "aura"]
    dataset_path = "dataset"
    for class_name in class_list:
        source_directory = f"D:/water-bottles-dataset/{class_name}"
        destination_directory = f"{dataset_path}/{class_name}"
        movefile_with_rename_and_convert(source_directory, destination_directory, class_name)
        

