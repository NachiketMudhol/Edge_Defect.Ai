import torch
from torchvision import models
import torch.nn as nn

# ---------------------------
# CONFIG
# ---------------------------
NUM_CLASSES = 8                 # change only if your classes differ
MODEL_PATH = "best_model.pth"
ONNX_PATH = "defect_model_fp32.onnx"
INPUT_SIZE = (1, 3, 224, 224)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---------------------------
# EXPORT ONNX (FP32)
# ---------------------------
dummy_input = torch.randn(INPUT_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True
)

print("✅ FP32 ONNX model exported:", ONNX_PATH)