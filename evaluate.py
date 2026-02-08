import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# TRANSFORMS (NO AUGMENTATION)
# ---------------------------
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# TEST DATASET
# ---------------------------
test_dataset = datasets.ImageFolder(
    "data_split/test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class_names = test_dataset.classes
print("Classes:", class_names)

# ---------------------------
# MODEL
# ---------------------------
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------------------
# EVALUATION
# ---------------------------
all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"\n✅ TEST ACCURACY: {test_accuracy:.2f}%")

# ---------------------------
# CLASSIFICATION REPORT
# ---------------------------
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Semiconductor Defect Classification")
plt.tight_layout()

# Save and show
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\n📁 Confusion matrix saved as confusion_matrix.png")