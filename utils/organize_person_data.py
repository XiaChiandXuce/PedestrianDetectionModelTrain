import os
import shutil

def organize_person_dataset(src_img, src_lbl, dst_base, max_files=2000):
    dst_img = os.path.join(dst_base, 'images')
    dst_lbl = os.path.join(dst_base, 'labels')

    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    # 前 2000 张图像
    img_files = sorted(os.listdir(src_img))[:max_files]
    for file in img_files:
        shutil.copyfile(os.path.join(src_img, file), os.path.join(dst_img, file))

    # 前 2000 个标签
    lbl_files = sorted(os.listdir(src_lbl))[:max_files]
    for file in lbl_files:
        shutil.copyfile(os.path.join(src_lbl, file), os.path.join(dst_lbl, file))

    print(f"✅ Person 数据已整理完毕，共复制了 {len(img_files)} 张图像和 {len(lbl_files)} 个标签。")

if __name__ == '__main__':
    organize_person_dataset(
        src_img=r"D:\备份资料\CV数据集\Roboflow数据集\People Detection.v9i.yolov8\train\images",
        src_lbl=r"D:\备份资料\CV数据集\Roboflow数据集\People Detection.v9i.yolov8\train\labels",
        dst_base=r"D:\备份资料\develop\Pythoncode\yolov8_pedestrian_train\data\dataset_person",
        max_files=2000
    )
