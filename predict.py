from ultralytics import YOLO

from config import dataset_path, load_config, project_path, select_device, weights_path


def main():
    config = load_config()
    prediction = config["prediction"]

    model = YOLO(str(weights_path(config)))
    model.predict(
        source=str(dataset_path(config) / prediction["source"]),
        imgsz=config["dataset"]["image_size"],
        conf=prediction["confidence"],
        iou=prediction["iou"],
        device=select_device(config),
        save=prediction["save"],
        save_txt=prediction["save_txt"],
        project=str(project_path(prediction["output_dir"])),
        name=prediction["name"],
        exist_ok=prediction["exist_ok"],
    )


if __name__ == "__main__":
    main()
