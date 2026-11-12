from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

WEIGHTS = ROOT / "runs" / "detect" / "yolo11n_obsea" / "weights" / "best.pt"

SOURCE = r"D:\04_Datasets\obsea_split_YOLO\23sp_4120img_34945annots_2688res\test\images"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():

    model = YOLO(str(WEIGHTS))

    model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=DEVICE,
        save=True,
        save_txt=True,
        project=str(ROOT / "runs" / "prediction"),
        name="yolo11n_obsea",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
