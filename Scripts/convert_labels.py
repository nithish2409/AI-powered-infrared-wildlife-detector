import os
import xml.etree.ElementTree as ET

# --- Configuration ---

# This list MUST be in the same order as the one in your wildlife_dataset.yaml
CLASS_NAMES = [
    'AmurTiger', 'Badger', 'BlackBear', 'Cow', 'Dog', 'Hare', 'Leopard',
    'LeopardCat', 'MuskDeer', 'RaccoonDog', 'RedFox', 'RoeDeer', 'Sable',
    'SikaDeer', 'Weasel', 'WildBoar', 'Y.T.Marten'
]

# Paths to the label directories
LABEL_DIRS = ['../datasets/labels/train/', '../datasets/labels/val/']

# --- End Configuration ---


def convert_xml_to_yolo(xml_file):
    """
    Converts a single PASCAL VOC XML file to a YOLO format .txt file.
    The new .txt file is created in the same directory.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            print(f"Warning: 'size' tag not found in {xml_file}. Skipping.")
            return

        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_NAMES:
                print(f"Warning: Class '{class_name}' not in CLASS_NAMES list. Skipping.")
                continue
            
            class_id = CLASS_NAMES.index(class_name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            dw = 1.0 / img_width
            dh = 1.0 / img_height

            x_center_norm = x_center * dw
            y_center_norm = y_center * dh
            width_norm = width * dw
            height_norm = height * dh

            yolo_lines.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")

        # Write the YOLO format .txt file
        txt_filepath = os.path.splitext(xml_file)[0] + '.txt'
        with open(txt_filepath, 'w') as f:
            f.write('\n'.join(yolo_lines))

    except Exception as e:
        print(f"Error processing file {xml_file}: {e}")

def main():
    """Main function to iterate through directories and convert all XML files."""
    for label_dir in LABEL_DIRS:
        if not os.path.exists(label_dir):
            print(f"Directory not found: {label_dir}. Skipping.")
            continue

        xml_files = [f for f in os.listdir(label_dir) if f.endswith('.xml')]
        print(f"\nFound {len(xml_files)} XML files in '{label_dir}'. Starting conversion...")

        for xml_file in xml_files:
            full_path = os.path.join(label_dir, xml_file)
            convert_xml_to_yolo(full_path)
            
        print(f"Conversion complete for '{label_dir}'.")
        # Optional: Verify by checking for .txt files
        txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} .txt label files in '{label_dir}'.")


if __name__ == '__main__':
    main()