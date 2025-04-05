from ultralytics import YOLO
import os

def main():
    # 加载预训练模型（可选 yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt）
    model = YOLO('yolov8s.pt')

    # 启动训练
    model.train(
        data='config/data_merged.yaml',  # 👈 你的混合数据配置文件
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,                        # 0 表示使用 GPU 0，如果你没有 GPU 可改为 'cpu'
        cache=True,                     # 加快训练速度
        name='merged_model',            # 模型子目录
        project='model_weights',        # 模型主目录
        val=True                        # 每轮进行验证
    )

    # 将 best.pt 重命名为 merged_model.pt 方便后续加载
    src = os.path.join('model_weights', 'merged_model', 'weights', 'best.pt')
    dst = os.path.join('model_weights', 'merged_model.pt')
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"✅ 模型训练完成，已保存为：{dst}")
    else:
        print("⚠️ 未找到 best.pt，请确认模型是否训练成功")

if __name__ == '__main__':
    main()
