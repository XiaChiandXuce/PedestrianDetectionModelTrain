from ultralytics import YOLO
import os

def main():
    model = YOLO('yolov8s.pt')  # 你也可以换成 yolov8n.pt（更轻量）

    model.train(
        data='config/data_vehicle.yaml',     # 车辆数据集的配置文件
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        cache=True,
        name='vehicle_model',                # 输出子目录
        project='model_weights',             # 输出主目录
        val=True
    )

    # 将 best.pt 重命名为 vehicle_model.pt（统一命名风格）
    src = os.path.join('model_weights', 'vehicle_model', 'weights', 'best.pt')
    dst = os.path.join('model_weights', 'vehicle_model.pt')
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"✅ 车辆模型已保存为 {dst}")

if __name__ == '__main__':
    main()
