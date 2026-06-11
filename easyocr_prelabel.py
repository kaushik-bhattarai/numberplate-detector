import argparse
import json
import os
import sys

IMAGE_DIR = "corrected-plates"
OUTPUT_PATH = "json_files/corrected_prelabels.json"
DEFAULT_LANGS = ["hi"]  #Devanagari (Nepali script)


def parse_langs(value):
    return [lang.strip() for lang in value.split(",") if lang.strip()]


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def resolve_device(use_gpu, force_cpu):
    if force_cpu:
        return False
    if use_gpu is not None:
        return use_gpu
    return cuda_available()


def get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        print("Warning: tqdm not installed. Run `pip install tqdm` for progress bars.", flush=True)
        return None


def load_reader_with_progress(langs, use_gpu, tqdm):
    """Load EasyOCR models with a spinner/progress bar."""
    stages = [
        "Initialising EasyOCR",
        "Downloading / loading detection model",
        "Downloading / loading recognition model",
        "Models ready",
    ]

    if tqdm is None:
        print(f"Loading EasyOCR models for {langs}...")
        print("(First run downloads models — this can take 1-2 minutes.)")
        import easyocr
        reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        print("Models ready.")
        return reader

    bar = tqdm(
        total=len(stages),
        desc="Loading models",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        file=sys.stdout,
    )

    def advance(label):
        bar.set_postfix_str(label, refresh=True)
        bar.update(1)

    advance(stages[0])
    import easyocr
    advance(stages[1])
    reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
    advance(stages[2])
    advance(stages[3])
    bar.close()
    return reader


def run_ocr_with_progress(reader, image_names, image_dir, tqdm):
    """Run OCR on all images, showing a per-image progress bar."""
    output = []

    if tqdm is None:
        total = len(image_names)
        for idx, img_name in enumerate(image_names, start=1):
            if idx == 1 or idx % 10 == 0 or idx == total:
                print(f"  [{idx}/{total}] {img_name}", flush=True)
            img_path = os.path.join(image_dir, img_name)
            output.append(_process_image(reader, img_name, img_path))
        return output

    bar = tqdm(
        image_names,
        desc="Running OCR",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} imgs [{elapsed}<{remaining}, {rate_fmt}]",
        file=sys.stdout,
    )
    for img_name in bar:
        bar.set_postfix_str(img_name[-28:], refresh=True)  # trim long names
        img_path = os.path.join(image_dir, img_name)
        output.append(_process_image(reader, img_name, img_path))

    return output


def _process_image(reader, img_name, img_path):
    result = reader.readtext(img_path)
    detections = [
        {
            "box": [[float(x), float(y)] for x, y in box],
            "text": text,
            "confidence": float(confidence),
        }
        for box, text, confidence in result
    ]
    return {"image": img_name, "detections": detections}


def save_with_progress(output, output_path, tqdm):
    if tqdm is None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return

    with tqdm(
        total=1,
        desc="Saving JSON",
        bar_format="{l_bar}{bar}| {elapsed}",
        file=sys.stdout,
    ) as bar:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        bar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run EasyOCR on cropped plate images and save raw detections."
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--langs",
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated EasyOCR language codes (e.g. en,hi)",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use GPU (default: auto-detect)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    args = parser.parse_args()

    langs = parse_langs(args.langs)
    use_gpu = resolve_device(args.gpu, args.cpu)
    tqdm = get_tqdm()

    image_names = sorted(
        name
        for name in os.listdir(args.image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )

    if not image_names:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Device : {'GPU' if use_gpu else 'CPU'}")
    if use_gpu:
        try:
            import torch
            print(f"GPU    : {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"Images : {len(image_names)}  |  Languages : {langs}")
    print()

    # ── Phase 1: load models ────────────────────────────────────────────────
    reader = load_reader_with_progress(langs, use_gpu, tqdm)
    print()

    # ── Phase 2: OCR ────────────────────────────────────────────────────────
    output = run_ocr_with_progress(reader, image_names, args.image_dir, tqdm)
    print()

    # ── Phase 3: save ───────────────────────────────────────────────────────
    save_with_progress(output, args.output, tqdm)

    total_detections = sum(len(r["detections"]) for r in output)
    print(f"\nDone. {len(output)} images, {total_detections} detections → {args.output}")


if __name__ == "__main__":
    main()