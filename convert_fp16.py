import onnx
from onnxconverter_common import float16

# ---------------------------
# PATHS (DO NOT OVERWRITE FP32)
# ---------------------------
FP32_MODEL_PATH = "defect_model_fp32.onnx"
FP16_MODEL_PATH = "defect_model_fp16.onnx"

# ---------------------------
# LOAD FP32 MODEL
# ---------------------------
model_fp32 = onnx.load(FP32_MODEL_PATH)

# ---------------------------
# CONVERT TO FP16
# ---------------------------
model_fp16 = float16.convert_float_to_float16(
    model_fp32,
    keep_io_types=True   # 🔑 keeps input/output stable
)

# ---------------------------
# SAVE FP16 MODEL
# ---------------------------
onnx.save(model_fp16, FP16_MODEL_PATH)

print("✅ FP16 ONNX model saved as:", FP16_MODEL_PATH)