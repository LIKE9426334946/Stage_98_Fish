
from pathlib import Path
import random

import cv2
import yaml
import matplotlib.pyplot as plt


# ==================== 配置 ====================

DATASET = Path(
    r"D:\04_Datasets\obsea_split_YOLO\23sp_4120img_34945annots_2688res"
)

SPLIT = "train"      # train / valid / test
NUM_IMAGES = 3       # 随机查看几张图片
SEED = 42


# ==================== 读取类别 ====================

with open(DATASET / "data.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CLASS_NAMES = config["names"]


# ==================== 可视化函数 ====================

def visualize(image_path):

    # 读取图片
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 复制一份用于绘制标注
    annotated = image.copy()

    height, width = image.shape[:2]

    # 查找对应的标签文件
    label_path = (
        DATASET / SPLIT / "labels" / f"{image_path.stem}.txt"
    )

    if label_path.exists():

        with open(label_path, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])

                # YOLO归一化坐标 -> 像素坐标
                x1 = int((xc - bw / 2) * width)
                y1 = int((yc - bh / 2) * height)
                x2 = int((xc + bw / 2) * width)
                y2 = int((yc + bh / 2) * height)

                # 取出类别名称
                if isinstance(CLASS_NAMES, dict):
                    class_name = CLASS_NAMES[class_id]
                else:
                    class_name = CLASS_NAMES[class_id]

                # 绘制矩形框
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # 绘制类别名称
                cv2.putText(
                    annotated,
                    class_name,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

    # 原图与标注图并排显示
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(annotated)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    fig.suptitle(image_path.name, fontsize=10)

    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================

def main():

    image_dir = DATASET / SPLIT / "images"

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise RuntimeError("没有找到图片")

    random.seed(SEED)

    selected = random.sample(
        image_paths,
        min(NUM_IMAGES, len(image_paths))
    )

    for image_path in selected:
        visualize(image_path)


if __name__ == "__main__":
    main()
