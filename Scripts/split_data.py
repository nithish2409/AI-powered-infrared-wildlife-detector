import os
import random
import shutil

# --- Configuration ---
# Path to the original dataset
ORIGINAL_IMAGES_DIR = '../Original_Dataset/Images/'
ORIGINAL_ANNOTATIONS_DIR = '../Original_Dataset/Annotations/'

# Path to the new structured dataset directory
BASE_OUTPUT_DIR = '../datasets/'

# Define the split ratio
# 80% for training, 20% for validation.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
# --- End Configuration ---

def split_dataset():
    """
    Splits the dataset into training and validation sets and organizes
    them into the required YOLO directory structure.
    """
    print("Starting dataset split...")

    # Create the necessary directories
    train_img_path = os.path.join(BASE_OUTPUT_DIR, 'images', 'train')
    val_img_path = os.path.join(BASE_OUTPUT_DIR, 'images', 'val')
    train_label_path = os.path.join(BASE_OUTPUT_DIR, 'labels', 'train')
    val_label_path = os.path.join(BASE_OUTPUT_DIR, 'labels', 'val')

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)
    print("Created new directory structure.")

    # Get all image filenames from the original directory
    all_filenames = [f for f in os.listdir(ORIGINAL_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_filenames)
    
    # Calculate split index
    split_index = int(len(all_filenames) * TRAIN_RATIO)

    # Divide filenames into training and validation sets
    train_filenames = all_filenames[:split_index]
    val_filenames = all_filenames[split_index:]

    print(f"Total images: {len(all_filenames)}")
    print(f"Training images: {len(train_filenames)}")
    print(f"Validation images: {len(val_filenames)}")

    # Function to copy files
    def copy_files(filenames, img_dest_path, label_dest_path):
        copied_count = 0
        for filename in filenames:
            base_filename = os.path.splitext(filename)[0]
            img_src = os.path.join(ORIGINAL_IMAGES_DIR, filename)
            label_src = os.path.join(ORIGINAL_ANNOTATIONS_DIR, base_filename + '.xml')

            if os.path.exists(img_src) and os.path.exists(label_src):
                shutil.copyfile(img_src, os.path.join(img_dest_path, filename))
                shutil.copyfile(label_src, os.path.join(label_dest_path, base_filename + '.xml'))
                copied_count += 1
            else:
                print(f"Warning: Could not find image or label for {base_filename}")
        print(f"Successfully copied {copied_count} file pairs.")

    print("\nCopying training files...")
    copy_files(train_filenames, train_img_path, train_label_path)
    
    print("\nCopying validation files...")
    copy_files(val_filenames, val_img_path, val_label_path)

    print("\nDataset splitting and copying complete!")

if __name__ == '__main__':
    split_dataset()