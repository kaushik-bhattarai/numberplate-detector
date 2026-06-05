import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm

MODEL_PATH  = "best_resnet_plate_chars.pt"
IMAGE_FOLDER = "/home/kaushik/Nepali_Num_plate/ocr-inference"      
OUTPUT_CSV  = "predictions.csv"             
TOP_K       = 3                             

CLASSES = [
    'क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ',
    'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा', 'भे',
    'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह',
    '०', '१', '२', '३', '४', '५', '६', '७', '८', '९'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_model(model_path: str, num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

#Predict a single image
def predict(model: nn.Module, image_path: str) -> dict:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"file": os.path.basename(image_path), "error": str(e)}

    tensor = transform(img).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]       # (num_classes,)

    top_probs, top_idxs = torch.topk(probs, k=TOP_K)

    return {
        "file"      : os.path.basename(image_path),
        "prediction": CLASSES[top_idxs[0].item()],
        "confidence": f"{top_probs[0].item() * 100:.2f}%",
        "top_k"     : [
            (CLASSES[i.item()], f"{p.item()*100:.2f}%")
            for i, p in zip(top_idxs, top_probs)
        ],
    }

#Run over folder
def run_folder(folder: str) -> list[dict]:
    image_paths = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])

    if not image_paths:
        print(f"No supported images found in: {folder}")
        return []

    print(f"Found {len(image_paths)} images in '{folder}'")
    model   = load_model(MODEL_PATH, len(CLASSES))
    results = []

    for path in tqdm(image_paths, desc="Predicting", unit="img",
                     colour="cyan", dynamic_ncols=True):
        result = predict(model, path)
        results.append(result)

        if "error" in result:
            tqdm.write(f"  [ERROR] {result['file']}: {result['error']}")
        else:
            top_str = "  |  ".join(f"{c} {p}" for c, p in result["top_k"])
            tqdm.write(f"  {result['file']:<30}  →  {top_str}")

    return results

def save_csv(results: list[dict], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["file", "prediction", "confidence"] + \
                 [f"top{i+1}_class" for i in range(TOP_K)] + \
                 [f"top{i+1}_conf"  for i in range(TOP_K)]
        writer.writerow(header)

        for r in results:
            if "error" in r:
                writer.writerow([r["file"], "ERROR", r["error"]])
                continue
            top_classes = [c for c, _ in r["top_k"]]
            top_confs   = [p for _, p in r["top_k"]]
            writer.writerow([r["file"], r["prediction"], r["confidence"],
                             *top_classes, *top_confs])

    print(f"\nPredictions saved to: {path}")

# ─────────────────────────────────────────────
# 7. CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nepali plate character inference")
    parser.add_argument("--folder",    default=IMAGE_FOLDER, help="Folder of images")
    parser.add_argument("--model",     default=MODEL_PATH,   help="Path to .pt weights")
    parser.add_argument("--output",    default=OUTPUT_CSV,   help="CSV output path (optional)")
    parser.add_argument("--topk",      default=TOP_K, type=int, help="Top-K predictions")
    args = parser.parse_args()

    # Allow CLI overrides
    MODEL_PATH    = args.model
    TOP_K         = args.topk

    results = run_folder(args.folder)

    if results:
        ok      = [r for r in results if "error" not in r]
        errors  = [r for r in results if "error" in r]
        print(f"\nSummary: {len(ok)} succeeded, {len(errors)} failed")

        if args.output:
            save_csv(results, args.output)