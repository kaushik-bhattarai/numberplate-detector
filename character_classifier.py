import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import time

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DATA_ROOT  = "/home/kaushik/Nepali_Num_plate/ocr_dataset"
BATCH_SIZE = 8
EPOCHS     = 20
LR         = 1e-4
SAVE_PATH  = "best_resnet_plate_chars.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05)
    ], p=0.7),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

val_dataset.dataset = datasets.ImageFolder(root=DATA_ROOT, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

num_classes = len(full_dataset.classes)
print(f"Device  : {DEVICE}")
print(f"Classes : {num_classes}  ->  {full_dataset.classes}")
print(f"Train   : {train_size} samples  |  Val: {val_size} samples\n")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

history  = {"train_loss": [], "val_loss": [], "val_acc": []}
best_acc = 0.0

# Outer bar: one tick per epoch
epoch_bar = tqdm(range(EPOCHS), desc="Overall", unit="epoch",
                 colour="cyan", dynamic_ncols=True)

for epoch in epoch_bar:
    epoch_start = time.time()

    # ── Train ──────────────────────────────────
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader,
                     desc=f"  E{epoch+1:02d}/{EPOCHS} Train",
                     leave=False, unit="batch",
                     colour="green", dynamic_ncols=True)

    for images, labels in train_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # ── Validate ───────────────────────────────
    model.eval()
    val_loss = 0.0
    correct  = 0
    total    = 0

    val_bar = tqdm(val_loader,
                   desc=f"  E{epoch+1:02d}/{EPOCHS} Val  ",
                   leave=False, unit="batch",
                   colour="yellow", dynamic_ncols=True)

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_bar.set_postfix(loss=f"{loss.item():.4f}",
                                acc=f"{100.0*correct/total:.2f}%")

    avg_val_loss = val_loss / len(val_loader)
    acc          = 100.0 * correct / total
    epoch_time   = time.time() - epoch_start

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(acc)

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(acc)
    new_lr  = optimizer.param_groups[0]["lr"]
    lr_tag  = f"  LR {prev_lr:.1e}->{new_lr:.1e}" if new_lr != prev_lr else ""

    saved_tag = ""
    if acc > best_acc:
        best_acc  = acc
        torch.save(model.state_dict(), SAVE_PATH)
        saved_tag = "  [SAVED]"

    # Update outer bar summary
    epoch_bar.set_postfix(
        tl=f"{avg_train_loss:.4f}",
        vl=f"{avg_val_loss:.4f}",
        acc=f"{acc:.2f}%",
        t=f"{epoch_time:.1f}s",
    )
    tqdm.write(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train: {avg_train_loss:.4f} | "
        f"Val: {avg_val_loss:.4f} | "
        f"Acc: {acc:.2f}% | "
        f"{epoch_time:.1f}s"
        f"{lr_tag}{saved_tag}"
    )

tqdm.write(f"\nDone. Best val accuracy: {best_acc:.2f}%")

epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss")
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_range, history["val_acc"], color="green", label="Val Accuracy")
axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()

all_preds  = []
all_labels = []

eval_bar = tqdm(val_loader, desc="Evaluating", unit="batch",
                colour="magenta", dynamic_ncols=True)

with torch.no_grad():
    for images, labels in eval_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ── Confusion matrix ──
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
disp.plot(cmap="Blues", xticks_rotation=90, values_format="d")
plt.title("Final Confusion Matrix (Character OCR)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── Per-class accuracy ──
print("\nPer-class accuracy:")
print(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc (%)':>10}")
print("-" * 42)
for i, cls in enumerate(full_dataset.classes):
    mask    = np.array(all_labels) == i
    correct = (np.array(all_preds)[mask] == i).sum()
    total   = mask.sum()
    acc_cls = 100.0 * correct / total if total > 0 else 0.0
    print(f"{cls:<12} {correct:>8} {total:>8} {acc_cls:>9.2f}%")