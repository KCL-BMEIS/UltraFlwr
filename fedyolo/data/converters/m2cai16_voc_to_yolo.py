import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil


def convert_bbox_to_yolo(size, box):
    """Convert VOC bbox to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    # VOC format: xmin, ymin, xmax, ymax
    x = (float(box[0]) + float(box[2])) / 2.0
    y = (float(box[1]) + float(box[3])) / 2.0
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])

    # YOLO format: x_center, y_center, width, height (normalized)
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    print(f"Converted bbox {box} to YOLO format: {(x, y, w, h)}")
    return (x, y, w, h)


def convert_annotation(xml_file, output_dir, class_mapping):
    """Convert single XML file to YOLO format."""
    print(f"Processing file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    assert size is not None, f"No size element found in {xml_file}"

    width_elem = size.find("width")
    height_elem = size.find("height")
    assert (
        width_elem is not None and width_elem.text is not None
    ), f"Invalid width in {xml_file}"
    assert (
        height_elem is not None and height_elem.text is not None
    ), f"Invalid height in {xml_file}"

    width = int(width_elem.text)
    height = int(height_elem.text)

    output_file = os.path.join(output_dir, Path(xml_file).stem + ".txt")

    with open(output_file, "w") as out_file:
        for obj in root.iter("object"):
            name_elem = obj.find("name")
            if name_elem is None or name_elem.text is None:
                print(f"Skipping object with invalid name in {xml_file}")
                continue

            class_name = name_elem.text
            if class_name not in class_mapping:
                print(f"Skipping unknown class: {class_name}")
                continue

            class_id = class_mapping[class_name]
            xmlbox = obj.find("bndbox")
            if xmlbox is None:
                print(f"Skipping object with no bndbox in {xml_file}")
                continue

            xmin_elem = xmlbox.find("xmin")
            ymin_elem = xmlbox.find("ymin")
            xmax_elem = xmlbox.find("xmax")
            ymax_elem = xmlbox.find("ymax")

            if any(
                elem is None or elem.text is None
                for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]
            ):
                print(f"Skipping object with invalid bounding box in {xml_file}")
                continue

            # Cast to str since we've verified they're not None
            xmin_text: str = xmin_elem.text  # type: ignore
            ymin_text: str = ymin_elem.text  # type: ignore
            xmax_text: str = xmax_elem.text  # type: ignore
            ymax_text: str = ymax_elem.text  # type: ignore

            b = (
                float(xmin_text),
                float(ymin_text),
                float(xmax_text),
                float(ymax_text),
            )
            bb = convert_bbox_to_yolo((width, height), b)

            out_file.write(
                f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n"
            )
            print(f"Written bbox for class {class_name} with id {class_id}: {bb}")


def create_data_yaml(output_path, class_mapping):
    """Create data.yaml file."""
    with open(output_path, "w") as f:
        f.write("train: ../train/images\n")
        f.write("val: ../valid/images\n")
        f.write("test: ../test/images\n\n")
        f.write(f"nc: {len(class_mapping)}\n")
        f.write(f"names: {list(class_mapping.keys())}\n")
    print(f"data.yaml created at {output_path}")


def read_class_mapping(file_path):
    """Read class mapping from txt file and convert to 0-based indexing."""
    class_mapping = {}
    with open(file_path, "r") as f:
        for line in f:
            idx, class_name = line.strip().split(" ", 1)
            # Convert to 0-based indexing for YOLO format
            class_mapping[class_name] = int(idx) - 1
    print(f"Class mapping read from {file_path}: {class_mapping}")
    return class_mapping


def copy_files(file_list, src_dir, dest_dir, file_extension):
    """Copy files from source to destination directory."""
    os.makedirs(dest_dir, exist_ok=True)
    for file_name in file_list:
        src_file = os.path.join(src_dir, file_name + file_extension)
        dest_file = os.path.join(dest_dir, file_name + file_extension)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
        else:
            print(f"File {src_file} does not exist and was skipped.")


def main():
    # Define paths
    xml_dir = "/home/user/projects/datasets/m2cai16-tool-locations/m2cai16-tool-locations/Annotations"
    output_dir = "./yolo_annotations"  # Output directory for YOLO format annotations
    class_list_path = "class_list.txt"  # Path to class list file
    image_dir = "/home/user/projects/datasets/m2cai16-tool-locations/m2cai16-tool-locations/JPEGImages"  # Update this to the path where images are stored

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Read class mapping from file
    class_mapping = read_class_mapping(class_list_path)

    # Convert all XML files
    for xml_file in glob.glob(os.path.join(xml_dir, "*.xml")):
        convert_annotation(xml_file, output_dir, class_mapping)

    # Create data.yaml
    create_data_yaml(os.path.join("./", "data.yaml"), class_mapping)
    print("Conversion completed successfully.")

    # Read train, val, and test splits
    with open("ImageSets/Main/train.txt") as f:
        train_files = [line.strip() for line in f]
    with open("ImageSets/Main/val.txt") as f:
        val_files = [line.strip() for line in f]
    with open("ImageSets/Main/test.txt") as f:
        test_files = [line.strip() for line in f]

    # Copy images and labels to respective directories
    copy_files(train_files, image_dir, "./yolo_data_split/train/images", ".jpg")
    copy_files(train_files, output_dir, "./yolo_data_split/train/labels", ".txt")
    copy_files(val_files, image_dir, "./yolo_data_split/val/images", ".jpg")
    copy_files(val_files, output_dir, "./yolo_data_split/val/labels", ".txt")
    copy_files(test_files, image_dir, "./yolo_data_split/test/images", ".jpg")
    copy_files(test_files, output_dir, "./yolo_data_split/test/labels", ".txt")


if __name__ == "__main__":
    main()
