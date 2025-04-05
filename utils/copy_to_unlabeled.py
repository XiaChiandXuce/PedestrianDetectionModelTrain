import os
import shutil

def copy_to_unlabeled(src_dirs, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    total = 0
    for src in src_dirs:
        for file in os.listdir(src):
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst_dir, file)
            if not os.path.exists(dst_file):  # 避免重复
                shutil.copyfile(src_file, dst_file)
                total += 1
    print(f"✅ 共复制 {total} 张图像到 dataset_unlabeled/images")

if __name__ == '__main__':
    person_images = r"D:\备份资料\develop\Pythoncode\yolov8_pedestrian_train\data\dataset_person\images"
    vehicle_images = r"D:\备份资料\develop\Pythoncode\yolov8_pedestrian_train\data\dataset_vehicle\images"
    unlabeled_images = r"D:\备份资料\develop\Pythoncode\yolov8_pedestrian_train\data\dataset_unlabeled\images"

    copy_to_unlabeled([person_images, vehicle_images], unlabeled_images)
