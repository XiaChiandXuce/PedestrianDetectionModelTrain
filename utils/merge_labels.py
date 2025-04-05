# utils/merge_labels.py
import os
from pathlib import Path
from shutil import copyfile

# ==== 配置路径（绝对路径）====
UNLABELED_IMAGE_DIR = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/data/dataset_unlabeled/images"
PERSON_LABEL_DIR = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/predictions/person/labels"
VEHICLE_LABEL_DIR = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/predictions/vehicle/labels"
MERGED_IMAGE_DIR = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/data/dataset_merged/images"
MERGED_LABEL_DIR = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/data/dataset_merged/labels"

# 创建目标文件夹
os.makedirs(MERGED_IMAGE_DIR, exist_ok=True)
os.makedirs(MERGED_LABEL_DIR, exist_ok=True)

def merge_labels():
    image_paths = list(Path(UNLABELED_IMAGE_DIR).glob("*.jpg"))
    print(f"🧱 正在合并 {len(image_paths)} 张图像的标签...")

    for img_path in image_paths:
        stem = img_path.stem
        merged_label_path = Path(MERGED_LABEL_DIR) / f"{stem}.txt"
        person_label_path = Path(PERSON_LABEL_DIR) / f"{stem}.txt"
        vehicle_label_path = Path(VEHICLE_LABEL_DIR) / f"{stem}.txt"

        with open(merged_label_path, 'w') as fout:
            # 写入 person 标签
            if person_label_path.exists():
                with open(person_label_path, 'r') as fin:
                    fout.write(fin.read())

            # 写入 vehicle 标签
            if vehicle_label_path.exists():
                with open(vehicle_label_path, 'r') as fin:
                    fout.write(fin.read())

        # 复制图片
        dst_img_path = Path(MERGED_IMAGE_DIR) / img_path.name
        copyfile(img_path, dst_img_path)

    print(f"✅ 合并完成，共生成 {len(image_paths)} 张图像和标签")

if __name__ == "__main__":
    merge_labels()
