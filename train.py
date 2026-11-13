from ultralytics import YOLO

from config import common_args, load_config, model_name, project_path


def main():
    config = load_config()
    training = config["training"]
    args = common_args(config)

    print(f"Device: {args['device']}")

    model = YOLO(model_name(config))
    model.train(
        **args,
        epochs=training["epochs"],
        lr0=training["learning_rate"],
        optimizer=training["optimizer"],
        patience=training["patience"],
        seed=training["seed"],
        deterministic=training["deterministic"],
        pretrained=training["pretrained"],
        save=training["save"],
        plots=training["plots"],
        project=str(project_path(training["output_dir"])),
        name=training["name"],
        exist_ok=training["exist_ok"],
    )


if __name__ == "__main__":
    main()
