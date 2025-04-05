from pathlib import Path

label_dir = Path(r"D:/备份资料/develop/Pythoncode/yolov8_pedestrian_train/data/dataset_merged/labels")

txt_files = list(label_dir.glob("*.txt"))
cleaned = 0

for file in txt_files:
    lines = file.read_text().strip().splitlines()
    cleaned_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # 只保留前 5 列（class_id, x, y, w, h）
            cleaned_lines.append(" ".join(parts[:5]))

    if cleaned_lines:
        file.write_text("\n".join(cleaned_lines) + "\n")
        cleaned += 1

print(f"✅ 清洗完成，共处理 {cleaned} 个标签文件，去除置信度列")
