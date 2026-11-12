
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DATA = r"D:\04_Datasets\obsea_split_YOLO\23sp_4120img_34945annots_2688res\data.yaml"

WEIGHTS = ROOT / "runs" / "detect" / "yolo11n_obsea" / "weights" / "best.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():

    model = YOLO(str(WEIGHTS))

    metrics = model.val(
        data=DATA,
        split="test",

        imgsz=640,
        batch=8,
        device=DEVICE,
        workers=0,

        plots=True,

        project=str(ROOT / "runs" / "evaluation"),
        name="test",
        exist_ok=True
    )

    print("\n测试集评估结果")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
