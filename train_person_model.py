from ultralytics import YOLO
import os

def main():
    model = YOLO('yolov8s.pt')  # 加载预训练模型（可换 yolov8n.pt）

    model.train(
        data='config/data_person.yaml',       # 只含 person 的数据配置
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        cache=True,
        name='person_model',                  # 模型输出子目录
        project='model_weights',              # 模型输出主目录
        val=True
    )

    # 自动重命名 best.pt 为你期望的名字（覆盖式保存）
    src = os.path.join('model_weights', 'person_model', 'weights', 'best.pt')
    dst = os.path.join('model_weights', 'person_model.pt')
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"✅ 模型已保存为 {dst}")

if __name__ == '__main__':
    main()
