import os
import xml.etree.ElementTree as ET
from collections import Counter

# --- Configuration ---
# Set the path to the directory containing your .xml annotation files
ANNOTATIONS_DIR = '../Original_Dataset/Annotations/' 
# --- End Configuration ---

def extract_and_count_classes(annotations_dir):
    """
    Parses all .xml files in a directory to find unique class names and
    count the instances of each.
    """
    class_counts = Counter()
    class_names = set()

    if not os.path.exists(annotations_dir):
        print(f"Error: Directory not found at '{annotations_dir}'")
        return None, None
        
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    if not xml_files:
        print(f"Error: No .xml files found in '{annotations_dir}'. Please check the path.")
        return None, None

    print(f"Found {len(xml_files)} XML files. Processing...")

    for filename in xml_files:
        try:
            tree = ET.parse(os.path.join(annotations_dir, filename))
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name:
                    class_names.add(class_name)
                    class_counts[class_name] += 1
        except ET.ParseError:
            print(f"Warning: Skipping malformed XML file: {filename}")

    # Return a sorted list of unique names and the Counter object
    return sorted(list(class_names)), class_counts

def main():
    unique_names, counts = extract_and_count_classes(ANNOTATIONS_DIR)

    if not unique_names:
        print("\nCould not find any classes. Please check your configuration.")
        return

    # --- 1. Print Instance Counts ---
    print("\n--- Instance Counts per Class ---")
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    for name, count in sorted_counts.items():
        print(f"{name:<20}: {count}")

    # --- 2. Print YAML-formatted Class List ---
    print("\n" + "="*40)
    print("COPY AND PASTE THIS INTO YOUR .yaml FILE")
    print("="*40 + "\n")
    print(f"nc: {len(unique_names)}")
    print("names:")
    for name in unique_names:
        print(f"  - '{name}'")
    print("\n" + "="*40)

if __name__ == '__main__':
    main()