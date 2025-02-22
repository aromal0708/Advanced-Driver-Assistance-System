import os
import pandas as pd
import shutil
from tqdm import tqdm

# Define dataset paths

# Get the absolute path of the project root dynamically
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  #gets the path of the base directory dynamically form the utils folder

# BASE_DIR = "datasets/traffic_sign/"  # =>if its not working in the utils folder

# Define dataset paths using the root directory
BASE_DIR = os.path.join(ROOT_DIR, "datasets/traffic_sign/")
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "raw/GTSRB/Final_Training/Images/")
RAW_TEST_DIR = os.path.join(BASE_DIR, "raw/")

ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations/")
IMAGES_DIR = os.path.join(BASE_DIR, "images/")
LABELS_DIR = os.path.join(BASE_DIR, "labels/")

# Create necessary directories
for folder in [
    os.path.join(IMAGES_DIR, "train/"),
    os.path.join(IMAGES_DIR, "test/"),
    os.path.join(LABELS_DIR, "train/"),
    os.path.join(LABELS_DIR, "test/"),
    os.path.join(ANNOTATIONS_DIR, "train/"),
    os.path.join(ANNOTATIONS_DIR, "test/")
]:
    os.makedirs(folder, exist_ok=True)

# Step 1: Move Train Images & Annotations
print("Organizing training images and annotations...")

for class_folder in tqdm(os.listdir(RAW_TRAIN_DIR)):
    class_path = os.path.join(RAW_TRAIN_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue  # Skip non-folder files

    # Move annotation CSV
    gt_file = os.path.join(class_path, f"GT-{class_folder}.csv")
    shutil.copy(gt_file, os.path.join(ANNOTATIONS_DIR, "train/"))

    # Read annotation file
    df = pd.read_csv(gt_file, sep=";")

    for _, row in df.iterrows():
        img_name = row["Filename"]
        img_path = os.path.join(class_path, img_name)

        # Move image to images/train/
        shutil.copy(img_path, os.path.join(IMAGES_DIR, "train/", img_name))

        # Convert annotation to YOLO format
        img_w, img_h = row["Width"], row["Height"]
        x_center = (row["Roi.X1"] + row["Roi.X2"]) / 2 / img_w
        y_center = (row["Roi.Y1"] + row["Roi.Y2"]) / 2 / img_h
        w = (row["Roi.X2"] - row["Roi.X1"]) / img_w
        h = (row["Roi.Y2"] - row["Roi.Y1"]) / img_h
        label_txt = f"{row['ClassId']} {x_center} {y_center} {w} {h}\n"

        # Save label file
        label_file = os.path.join(LABELS_DIR, "train/", img_name.replace(".ppm", ".txt"))
        with open(label_file, "w") as f:
            f.write(label_txt)

# Step 2: Move Test Images & Annotations
print("\U0001F4E6 Organizing test images and annotations...")

shutil.copy(TEST_ANNOTATION_FILE, os.path.join(ANNOTATIONS_DIR, "test/"))  # Corrected test annotation path

df_test = pd.read_csv(TEST_ANNOTATION_FILE, sep=";")

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    img_name = row["Filename"]
    img_path = os.path.join(RAW_TEST_DIR, img_name)  # Corrected test image path

    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found!")  # Added warning for missing files
        continue

    # Move image to images/test/
    shutil.copy(img_path, os.path.join(IMAGES_DIR, "test/", img_name))

    # Convert annotation to YOLO format
    img_w, img_h = row["Width"], row["Height"]
    x_center = (row["Roi.X1"] + row["Roi.X2"]) / 2 / img_w
    y_center = (row["Roi.Y1"] + row["Roi.Y2"]) / 2 / img_h
    w = (row["Roi.X2"] - row["Roi.X1"]) / img_w
    h = (row["Roi.Y2"] - row["Roi.Y1"]) / img_h
    label_txt = f"{row['ClassId']} {x_center} {y_center} {w} {h}\n"

    # Save label file
    label_file = os.path.join(LABELS_DIR, "test/", img_name.replace(".ppm", ".txt"))
    with open(label_file, "w") as f:
        f.write(label_txt)

# Step 3: Create data.yaml file
print("\U0001F4E6 Creating data.yaml for YOLO training...")

classes = [str(i) for i in range(43)]  # 43 classes in GTSRB
yaml_content = f"""train: {IMAGES_DIR}train/
val: {IMAGES_DIR}test/
nc: 43
names: {classes}
"""

with open(os.path.join(BASE_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("\U0001F389 Dataset is organized and ready for training!")
