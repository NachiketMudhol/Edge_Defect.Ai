import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "defect_model_fp16.onnx"   # 🔑 FP16 MODEL
DATASET_PATH = "data_split/test"
BATCH_SIZE = 1

# ---------------------------
# TRANSFORMS (MATCH TRAINING)
# ---------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# DATASET
# ---------------------------
test_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes

print("Classes:", class_names)

# ---------------------------
# ONNX RUNTIME SESSION
# ---------------------------
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# ---------------------------
# INFERENCE
# ---------------------------
all_preds = []
all_labels = []

for images, labels in test_loader:
    images_np = images.numpy()  # FP32 input (correct)
    outputs = session.run(None, {input_name: images_np})[0]
    preds = np.argmax(outputs, axis=1)

    all_preds.extend(preds)
    all_labels.extend(labels.numpy())

# ---------------------------
# METRICS
# ---------------------------
acc = accuracy_score(all_labels, all_preds) * 100
print(f"\n✅ FP16 ONNX TEST ACCURACY: {acc:.2f}%\n")

print("📊 CLASSIFICATION REPORT:")
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
plt.title("Confusion Matrix – FP16 ONNX Model")
plt.tight_layout()
plt.savefig("confusion_matrix_fp16.png", dpi=300)
plt.show()

print("\n📁 Confusion matrix saved as confusion_matrix_fp16.png")