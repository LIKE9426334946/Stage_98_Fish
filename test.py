from ultralytics import YOLO

from config import common_args, load_config, project_path, weights_path


def main():
    config = load_config()
    evaluation = config["evaluation"]

    model = YOLO(str(weights_path(config)))
    metrics = model.val(
        **common_args(config),
        split=evaluation["test_split"],
        plots=evaluation["plots"],
        project=str(project_path(evaluation["output_dir"])),
        name=evaluation["test_name"],
        exist_ok=evaluation["exist_ok"],
    )

    print("\n测试集评估结果")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
