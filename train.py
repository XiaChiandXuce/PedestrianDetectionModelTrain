from ultralytics import YOLO

def main():
    # 加载预训练模型（你可以选择 yolov8n.pt 更轻量）
    model = YOLO('yolov8s.pt')

    # 开始训练
    model.train(
        data='config/data.yaml',         # 数据集配置文件路径
        epochs=100,                      # 训练轮数
        imgsz=640,                       # 输入图片大小
        batch=8,                         # 批大小（默认16，这里减小）
        workers=2,                       # 数据加载线程数（默认8，这里减小）
        device=0,                        # 使用GPU 0
        cache=True,                      # 加速数据加载，缓存图片
        name='yolov8_pedestrian_vehicle3', # 输出目录名
        val=True                         # 开启验证集评估
    )

if __name__ == '__main__':
    # 多进程训练时必须加这个保护
    main()
