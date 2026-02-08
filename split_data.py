import os
import shutil
import random

source_dir = "data/processed"
base_dir = "data_split"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    train_end = int(len(images) * train_ratio)
    val_end = int(len(images) * (train_ratio + val_ratio))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        split_path = os.path.join(base_dir, split, cls)
        os.makedirs(split_path, exist_ok=True)
        for img in splits[split]:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(split_path, img))

print("Dataset split complete!")
