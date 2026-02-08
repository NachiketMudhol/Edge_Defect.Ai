import os
import cv2

input_dir = "data/raw"
output_dir = "data/processed"
img_size = 224

classes = os.listdir(input_dir)

for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    for img_name in os.listdir(os.path.join(input_dir, cls)):
        img_path = os.path.join(input_dir, cls, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join(output_dir, cls, img_name), img)

print("✅ Preprocessing Done")