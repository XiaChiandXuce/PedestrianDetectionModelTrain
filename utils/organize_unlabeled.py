from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm

# ✅ 绝对路径配置（请根据你项目实际路径确认）
UNLABELED_IMAGE_DIR = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/data/dataset_unlabeled/images"
PERSON_LABEL_DIR    = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/predictions/person/labels"
VEHICLE_LABEL_DIR   = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/predictions/vehicle/labels"
PERSON_MODEL_PATH   = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/model_weights/person_model.pt"
VEHICLE_MODEL_PATH  = r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/model_weights/vehicle_model.pt"

CONF_THRESHOLD = 0.4

# ✅ 创建输出目录
os.makedirs(PERSON_LABEL_DIR, exist_ok=True)
os.makedirs(VEHICLE_LABEL_DIR, exist_ok=True)

def predict_and_save(model, class_offset, image_paths, label_dir):
    for img_path in tqdm(image_paths, desc=f"推理中 ({label_dir})"):
        img = Image.open(img_path)
        width, height = img.size

        results = model(img_path, conf=CONF_THRESHOLD)
        boxes = results[0].boxes

        label_path = Path(label_dir) / (Path(img_path).stem + ".txt")
        with open(label_path, "w") as f:
            for box in boxes:
                cls = int(box.cls[0]) + class_offset
                xywh = box.xywh[0].tolist()
                conf = box.conf[0].item()

                # ✅ 归一化坐标
                x, y, w, h = xywh
                x /= width
                y /= height
                w /= width
                h /= height

                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

def main():
    # ✅ 路径存在性校验
    assert os.path.exists(PERSON_MODEL_PATH), "❌ PERSON_MODEL_PATH 不存在"
    assert os.path.exists(VEHICLE_MODEL_PATH), "❌ VEHICLE_MODEL_PATH 不存在"
    assert os.path.isdir(UNLABELED_IMAGE_DIR), "❌ UNLABELED_IMAGE_DIR 不存在"

    print("🚶 加载行人模型...")
    person_model = YOLO(PERSON_MODEL_PATH)

    print("🚗 加载车辆模型...")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    # ✅ 获取所有图像路径
    image_paths = list(Path(UNLABELED_IMAGE_DIR).glob("*.jpg"))
    print(f"📷 共检测到 {len(image_paths)} 张图片")

    print("🧠 使用行人模型进行推理...")
    predict_and_save(person_model, class_offset=0, image_paths=image_paths, label_dir=PERSON_LABEL_DIR)

    print("🧠 使用车辆模型进行推理...")
    predict_and_save(vehicle_model, class_offset=1, image_paths=image_paths, label_dir=VEHICLE_LABEL_DIR)

    print("✅ 所有标签保存完成，已分别存入 person/labels 和 vehicle/labels")

if __name__ == "__main__":
    main()
