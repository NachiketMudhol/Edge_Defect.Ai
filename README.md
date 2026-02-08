# Edge_Defect.Ai
Overview
Edge-Defect-AI is a lightweight Edge-AI semiconductor defect classification system optimized for CPU-based edge devices. 
It uses a MobileNetV3-Small model and FP16 optimization to achieve high accuracy with a small model size.

What this project does
- Classifies semiconductor defect images into 8 categories
- Uses MobileNetV3-Small for efficient inference
- Optimized from FP32 to FP16 for smaller size and faster edge performance
- Designed for real-time edge deployment

Final Model
- Model file: defect_model_fp16.onnx
- Size: ~3 MB
- Accuracy: ~98–99%
- Framework: PyTorch → ONNX → FP16

Project Pipeline (Actual Order)
1. Preprocess raw images
2. Split dataset into train/val/test
3. Train MobileNetV3 model
4. Evaluate trained model (FP32)
5. Export model to ONNX (FP32)
6. Convert FP32 ONNX to FP16
7. Evaluate FP16 model
8. Generate confusion matrix

How to Reproduce
Run the scripts in this order:

python preprocess.py
python split_data.py
python train.py
python evaluate.py
python export_onnx.py
python convert_fp16.py
python evaluate_fp16_onnx.py

Key Results
FP32 Model:
- Size: ~6.3 MB
- Accuracy: ~99%

FP16 Model:
- Size: ~3.1 MB
- Accuracy: ~98%

Result:
- ~50% reduction in model size
- Minimal accuracy drop
- Faster edge inference

