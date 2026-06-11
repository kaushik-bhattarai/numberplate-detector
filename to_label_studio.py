import argparse
import json
import os
import uuid

from PIL import Image

DEFAULT_INPUT = "json_files/corrected_prelabels.json"
DEFAULT_OUTPUT = "json_files/corrected_label_studio_tasks.json"
DEFAULT_IMAGE_DIR = "corrected-plates"
DEFAULT_MODEL_VERSION = "easyocr"


def box_to_percentages(box, img_w, img_h):
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return {
        "x": x_min / img_w * 100,
        "y": y_min / img_h * 100,
        "width": (x_max - x_min) / img_w * 100,
        "height": (y_max - y_min) / img_h * 100,
    }


def detection_to_results(detection, img_w, img_h):
    region = box_to_percentages(detection["box"], img_w, img_h)
    region_id = uuid.uuid4().hex[:8]
    text = detection["text"]

    return [
        {
            "id": region_id,
            "from_name": "bbox",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                **region,
                "rotation": 0,
                "rectanglelabels": ["text"],
            },
        },
        {
            "id": region_id,
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "value": {
                **region,
                "rotation": 0,
                "text": [text],
            },
        },
    ]


def build_image_url(img_name, image_mode, local_files_subdir, image_base_url):
    if image_mode == "local-files":
        return f"/data/local-files/?d={local_files_subdir}/{img_name}"

    if image_mode == "http":
        if not image_base_url:
            raise ValueError("--image-base-url is required when --image-mode=http")
        return f"{image_base_url.rstrip('/')}/{local_files_subdir}/{img_name}"

    raise ValueError(f"Unsupported image mode: {image_mode}")


def convert(
    input_path,
    output_path,
    image_dir,
    local_files_subdir,
    as_annotations,
    model_version,
    image_mode,
    image_base_url,
):
    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    tasks = []
    target_key = "annotations" if as_annotations else "predictions"

    for item in raw_data:
        img_name = item["image"]
        img_path = os.path.join(image_dir, img_name)

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        result_list = []
        for detection in item.get("detections", []):
            result_list.extend(detection_to_results(detection, img_w, img_h))

        image_url = build_image_url(
            img_name, image_mode, local_files_subdir, image_base_url
        )
        entry = {
            "data": {"image": image_url},
            target_key: [{
                "model_version": model_version,
                "result": result_list,
            }],
        }
        tasks.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(tasks)} images -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert EasyOCR prelabels to Label Studio import JSON."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--local-files-subdir",
        default=DEFAULT_IMAGE_DIR,
        help="Path relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
    )
    parser.add_argument(
        "--as-annotations",
        action="store_true",
        help="Write editable annotations instead of read-only predictions",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help="Value stored in Label Studio model_version field",
    )
    parser.add_argument(
        "--image-mode",
        choices=["local-files", "http"],
        default="local-files",
        help="How image paths are written into task JSON",
    )
    parser.add_argument(
        "--image-base-url",
        default="",
        help="Base URL when --image-mode=http (e.g. http://127.0.0.1:8888)",
    )
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        image_dir=args.image_dir,
        local_files_subdir=args.local_files_subdir,
        as_annotations=args.as_annotations,
        model_version=args.model_version,
        image_mode=args.image_mode,
        image_base_url=args.image_base_url,
    )


if __name__ == "__main__":
    main()
