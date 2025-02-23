from PIL import Image
import os
import glob


def convert_ppm_to_jpg(folder):
    for file in glob.glob(f"{folder}/*.ppm"):  # Find all .ppm files
        img = Image.open(file)
        new_file = file.replace(".ppm", ".jpg")  # Change extension
        img.convert("RGB").save(new_file, "JPEG")  # Save as JPG
        os.remove(file)  # Remove the old .ppm file
        print(f"Converted {file} -> {new_file}")


# Update paths to your dataset directories
train_folder = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Advanced-Driver-Assistance-System/datasets/traffic_sign/images/train"
val_folder = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Advanced-Driver-Assistance-System/datasets/traffic_sign/images/test"

# Convert train and val images
convert_ppm_to_jpg(train_folder)
convert_ppm_to_jpg(val_folder)
