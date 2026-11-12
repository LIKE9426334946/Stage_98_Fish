
from pathlib import Path

import torch
from ultralytics import YOLO


# 项目根目录
ROOT = Path(__file__).resolve().parent

# 数据集配置文件
DATA = r"D:\04_Datasets\obsea_split_YOLO\23sp_4120img_34945annots_2688res\data.yaml"

# 自动选择设备
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# 训练配置
EPOCHS = 100
BATCH_SIZE = 8
IMAGE_SIZE = 640


def main():

    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载预训练模型
    model = YOLO("yolo11n.pt")

    # 开始训练
    model.train(
        data=DATA,

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=0,

        optimizer="auto",
        patience=20,

        seed=42,
        deterministic=True,

        pretrained=True,
        save=True,
        plots=True,

        project=str(ROOT / "runs" / "detect"),
        name="yolo11n_obsea",
        exist_ok=True
    )


if __name__ == "__main__":
    main()
