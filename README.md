# **🧠 PedestrianDetectionModelTrain - 模型训练子系统**

> **"一个好的 README，不是告诉你代码怎么运行，而是告诉你为什么这样设计。"**  
> —— **徐策**  
>  
> **"如果 README 不能让人 5 分钟内理解系统核心，那它就是失败的。"**  
> —— **夏驰**

---

## **📌 项目介绍**

本项目为 **《基于深度学习的道路行人检测系统的设计与实现》** 的核心训练子系统，专注于自定义 YOLOv8 行人检测模型训练，服务于主系统的 **行人检测、碰撞预警、复杂背景适应、姿态不变性** 等关键功能。

该子系统将基于自采集+自标注数据集进行从零训练，结合未来 10 篇参考论文中的技术路线逐步优化检测精度与鲁棒性，为主系统部署提供精准的权重支持。

---

## **📌 目录结构**
```plaintext
PedestrianDetectionModelTrain/
│
├── config/                  # ✅ 数据集配置文件（.yaml）
│   ├── data_person.yaml
│   ├── data_vehicle.yaml
│   └── data_merged.yaml
│
├── data/                    # ✅ 数据集目录
│   ├── dataset_person/      # 行人数据
│   ├── dataset_vehicle/     # 车辆数据
│   ├── dataset_merged/      # 合并数据集
│   └── dataset_unlabeled/   # 未标注图像（用于伪标签、半监督等）
│
├── model_weights/           # ✅ 模型权重输出
│   ├── person_model.pt
│   ├── vehicle_model.pt
│   └── merged_model.pt
│
├── predictions/             # ✅ 推理结果保存目录
│   ├── person/labels
│   └── vehicle/labels
│
├── utils/                   # ✅ 辅助工具脚本
│   ├── organize_person_data.py
│   ├── organize_vehicle_data.py
│   ├── copy_to_unlabeled.py
│   ├── merge_labels.py
│   └── logger.py
│
├── train_person_model.py     # 👤 训练行人检测模型
├── train_vehicle_model.py    # 🚗 训练车辆检测模型
├── train_merged_model.py     # 🧩 训练融合模型（人+车）
├── requirements.txt          # 依赖列表
└── README.md                 # 本文档
```

---

## **📌 系统功能映射**

本训练子系统对应主系统以下功能模块：

| 主系统功能项             | 训练模块支撑情况             |
|--------------------------|------------------------------|
| ✅ 行人检测              | 已完成 YOLOv8 自定义训练     |
| 🕐 多尺度检测            | 支持 `multi_scale=True`，待激活 |
| 🕐 姿态不变性识别        | 后续加入姿态分类辅助分支     |
| ✅ 复杂背景适应          | 数据集中覆盖建筑/树木/人群等 |
| ✅ 模型动态部署           | 支持主系统热切换权重文件     |
| 🕐 行人-车辆融合检测     | 已实现 `merged_model.pt` 训练 |
| ✅ 权重路径规范           | 与主系统约定路径自动接入     |

---

## **📌 快速开始**

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

建议使用 `conda` 或 `venv` 创建虚拟环境。

---

### 2️⃣ 开始训练

#### 👤 训练行人检测模型
```bash
python train_person_model.py
```

#### 🚗 训练车辆检测模型
```bash
python train_vehicle_model.py
```

#### 🧩 训练融合检测模型
```bash
python train_merged_model.py
```

---

## **📌 模型训练特点**

| 特性                    | 描述说明 |
|-------------------------|----------|
| ✅ 完全自定义数据集      | 不依赖 COCO，训练更贴近真实场景 |
| ✅ 多模型协同训练        | person / vehicle / merged 分任务训练 |
| ✅ 权重可直接部署         | `.pt` 权重可供 PyQt 主系统直接接入 |
| 🕐 多尺度训练支持        | `multi_scale=True` 参数可一键激活 |
| 🕐 半监督伪标签机制      | 已准备 `dataset_unlabeled/` 目录 |
| 🧱 支持边缘部署结构转换   | 后续导出 ONNX / TensorRT 格式 |

---

## **📌 论文优化方向（待逐步实现）**

本项目参考了以下 10 篇论文，后续将围绕这些方向持续优化模型性能：

1. ✅ **YOLOv8-Large 结构替换**（提升检测精度）
2. 🧠 **DCGAN 数据增强**（远距离小目标增强）
3. 🔀 **多网络融合机制**（Fused DNN）
4. 🧠 **深度强化学习策略**（DRL 场景自适应）
5. 💡 **亮度感知机制**（夜晚检测适应）
6. 🚨 **碰撞预警规则优化**（减少误报）
7. 🔍 **多尺度特征提取优化**（小目标更强）
8. 🎯 **目标跟踪 + ReID**（持续跟踪行人）
9. 🧩 **边缘计算部署优化**（适配 OrangePi 等）
10. 📷 **全景鱼眼检测支持**（畸变校正+多视角处理）

---

## **📌 项目定位与未来扩展**

本训练系统是整个智能交通行人检测平台的核心组件之一，未来将：

- 与 UI 系统无缝对接，实现热加载权重切换；
- 支持模型调参、验证可视化、检测视频保存；
- 推出增强版 train_eval 分析模块，集成 TensorBoard；
- 接入 NAS 网络结构搜索，进一步提升鲁棒性；
- 实现远程模型推理 + 设备下发机制。

---

## **📌 作者信息**

| 姓名 | 专业 | 项目时间 |
|------|------|-----------|
| 唐英昊 | 计算机科学与技术 | 2024.12 - 2025.6 |
| 指导教师 | 张蕊 教授 | 国家自然科学基金支持 |

---

## **📌 致谢**

感谢以下资源与支持：

- 🧠 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 提供强大目标检测框架  
- 🧾 所有参考论文作者 提供技术灵感

---

> **“不止于检测，更关注如何部署与落地。”**  
> —— 项目主张

```
