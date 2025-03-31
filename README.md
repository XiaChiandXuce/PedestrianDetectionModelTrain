# 🚶‍♂️ YOLOv8 行人检测训练子系统

本项目为毕业设计《基于深度学习的道路行人检测系统的设计与实现》的子模块，主要用于：

- 🎯 训练专属的 YOLOv8 行人检测模型
- 🏞️ 适配复杂背景、不同姿态、多尺度等实际道路场景
- 🔁 提供训练、验证、预测与模型导出的一站式流程

---

## ✅ 功能亮点

- 📦 支持自定义数据集（YOLO 格式）
- 🔧 支持训练参数调节与自动记录
- 📈 支持模型可视化验证与 ONNX 导出
- 🧠 后续支持多模型融合与迁移学习扩展

---

## 📂 项目结构

```plaintext
yolov8_pedestrian_train/
├── data/                    # 存放图像与标签
│   ├── images/              # 训练、验证、测试图像
│   └── labels/              # YOLO 格式标签（.txt）
│
├── config/
│   └── data.yaml            # 数据集配置文件（路径 + 类别）
│
├── models/                  # 保存训练生成的模型文件
│
├── utils/
│   └── logger.py            # 日志工具，用于记录训练过程
│
├── train.py                 # 主训练入口脚本
├── requirements.txt         # Python 依赖列表
└── README.md                # 项目说明文档
---
```

---
## ⚙️ 环境依赖

建议使用 **Python 3.9+** 和虚拟环境，推荐显卡 **RTX 3060** 或以上。

安装依赖：

```bash
pip install -r requirements.txt
```

---
