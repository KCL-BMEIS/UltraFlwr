import os
import json
import shutil
from pathlib import Path


def create_yolo_structure(data_home, output_dir):
    """
    Convert COCO annotations to YOLO format and structure the dataset.
    Also track and save:
      - Frames with annotations
      - Frames with no annotations (IDs saved to a file)
      - Duplicate bounding boxes
      - Annotation-image mismatches
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stats containers
    total_frames = {"train": 0, "val": 0, "test": 0}
    unique_frames = {
        "train": set(),
        "val": set(),
        "test": set(),
    }  # frames that have at least 1 bbox
    duplicates_count = {"train": 0, "val": 0, "test": 0}
    mismatches_count = {"train": 0, "val": 0, "test": 0}

    # Keep track of how many bounding-box annotations we process
    total_bboxes = {"train": 0, "val": 0, "test": 0}

    # We'll store frame IDs with no annotations at the end
    frames_no_annotations = {"train": [], "val": [], "test": []}

    # (image_id, class_id, x_center_rounded, y_center_rounded, w_rounded, h_rounded)
    seen_bboxes = {"train": set(), "val": set(), "test": set()}

    coco_categories = None  # Will be set from the last split's data

    for split in ["train", "val", "test"]:
        split_dir = Path(data_home) / split
        coco_json = split_dir / "annotation_coco.json"
        images_dir = Path(output_dir) / split / "images"
        labels_dir = Path(output_dir) / split / "labels"

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Load COCO annotations
        with open(coco_json, "r") as f:
            coco_data = json.load(f)

        # Store categories from the first split (all splits should have the same categories)
        if coco_categories is None:
            coco_categories = coco_data["categories"]

        # Count how many images (frames) are in this split
        total_frames[split] = len(coco_data["images"])

        # Create a category ID -> YOLO class ID mapping
        category_mapping = {cat["id"]: cat["id"] - 1 for cat in coco_data["categories"]}

        # Map images by their IDs for quick access
        image_info = {img["id"]: img for img in coco_data["images"]}

        # Process annotations
        for annotation in coco_data["annotations"]:
            image_id = annotation["image_id"]
            total_bboxes[split] += 1

            # Check if image_id is valid
            if image_id not in image_info:
                # This is an annotation->image mismatch
                mismatches_count[split] += 1
                continue

            # Retrieve the corresponding image
            image = image_info[image_id]
            img_file_name = image["file_name"]

            # Copy image to the target images directory (only if not already copied)
            source_image_path = split_dir / img_file_name
            target_image_path = images_dir / img_file_name
            if not target_image_path.exists():
                shutil.copy(source_image_path, target_image_path)

            # Generate YOLO labels
            img_width, img_height = image["width"], image["height"]
            x, y, w, h = annotation["bbox"]
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # Convert category_id to YOLO class_id
            class_id = category_mapping[annotation["category_id"]]

            # Mark that this frame has at least one bbox
            unique_frames[split].add(image_id)

            # Detect duplicates by rounding floats to 6 decimals
            bbox_sig = (
                image_id,
                class_id,
                round(x_center, 6),
                round(y_center, 6),
                round(w_norm, 6),
                round(h_norm, 6),
            )
            if bbox_sig in seen_bboxes[split]:
                duplicates_count[split] += 1
            else:
                seen_bboxes[split].add(bbox_sig)

            # Write YOLO label file
            label_file_name = Path(img_file_name).stem + ".txt"
            label_file_path = labels_dir / label_file_name
            with open(label_file_path, "a") as f_label:
                f_label.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                )

        # Find frames with no annotations in this split
        all_image_ids = set(image_info.keys())  # All image IDs in JSON
        no_annotation_ids = all_image_ids - unique_frames[split]

        # Save them as a sorted list
        frames_no_annotations[split] = sorted(list(no_annotation_ids))

        # Optionally, write them to a file
        no_ann_file = Path(output_dir) / f"no_annotations_{split}.txt"
        with open(no_ann_file, "w") as f_no_ann:
            for frame_id in frames_no_annotations[split]:
                f_no_ann.write(str(frame_id) + "\n")

    # Create data.yaml
    if coco_categories is not None:
        generate_data_yaml(
            output_dir,
            ["train", "val", "test"],
            coco_categories,
            total_frames,
            unique_frames,
            duplicates_count,
            total_bboxes,
            frames_no_annotations,
        )
    else:
        print("Warning: No categories found in COCO data")

    # Print summary
    print(f"Conversion complete! YOLO dataset saved in {output_dir}")
    print("\n===== Summary =====")
    for split in ["train", "val", "test"]:
        print(f"\nSplit: {split}")
        print(f"  - Total images in JSON:           {total_frames[split]}")
        print(f"  - Unique images with bounding boxes: {len(unique_frames[split])}")
        print(
            f"  - Images with no annotations:     {len(frames_no_annotations[split])}"
        )
        print(f"  - Total bounding-box annotations: {total_bboxes[split]}")


def generate_data_yaml(
    output_dir,
    splits,
    coco_categories,
    total_frames,
    unique_frames,
    duplicates_count,
    total_bboxes,
    frames_no_annotations,
):
    """
    Generate data.yaml file for YOLO-based frameworks using COCO category names.
    Also include stats about unique frames, duplicates, and bounding-box totals.
    """
    category_names = [cat["name"] for cat in coco_categories]

    yaml_content = {
        "train": str(Path(output_dir) / "train" / "images"),
        "val": str(Path(output_dir) / "val" / "images"),
        "test": str(Path(output_dir) / "test" / "images"),
        "nc": len(category_names),
        "names": category_names,
    }

    for split in splits:
        yaml_content[f"{split}_frames"] = total_frames[split]
        yaml_content[f"{split}_frames_with_bboxes"] = len(unique_frames[split])
        yaml_content[f"{split}_frames_no_annotations"] = len(
            frames_no_annotations[split]
        )
        yaml_content[f"{split}_duplicate_bboxes"] = duplicates_count[split]
        yaml_content[f"{split}_total_bboxes"] = total_bboxes[split]

    yaml_file = Path(output_dir) / "data.yaml"
    with open(yaml_file, "w") as f:
        for key, value in yaml_content.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    data_home = "<PATH_TO_DATA>"  # Example path
    output_dir = "endoscapes_yolo_format"
    create_yolo_structure(data_home, output_dir)
