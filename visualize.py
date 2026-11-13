import random

import cv2
import matplotlib.pyplot as plt
import yaml

from config import data_file, dataset_path, load_config


def visualize(image_path, label_dir, class_names, settings):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated = image.copy()
    height, width = image.shape[:2]
    label_path = label_dir / f"{image_path.stem}.txt"

    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])

                # YOLO 归一化坐标转换为像素坐标。
                x1 = int((xc - bw / 2) * width)
                y1 = int((yc - bh / 2) * height)
                x2 = int((xc + bw / 2) * width)
                y2 = int((yc + bh / 2) * height)

                color = tuple(settings["box_color"])
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    settings["box_thickness"],
                )
                cv2.putText(
                    annotated,
                    class_names[class_id],
                    (x1, max(y1 - settings["label_offset"], settings["label_min_y"])),
                    getattr(cv2, settings["font_face"]),
                    settings["font_scale"],
                    color,
                    settings["font_thickness"],
                    getattr(cv2, settings["line_type"]),
                )

    fig, axes = plt.subplots(1, 2, figsize=settings["figure_size"])
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(annotated)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    fig.suptitle(image_path.name, fontsize=settings["title_font_size"])
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    config = load_config()
    settings = config["visualization"]

    with data_file(config).open("r", encoding="utf-8") as file:
        class_names = yaml.safe_load(file)["names"]

    split_dir = dataset_path(config) / settings["split"]
    image_dir = split_dir / "images"
    extensions = {extension.lower() for extension in settings["image_extensions"]}
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        raise RuntimeError(f"没有找到图片: {image_dir}")

    selected = random.Random(settings["seed"]).sample(
        image_paths, min(settings["num_images"], len(image_paths))
    )
    for image_path in selected:
        visualize(image_path, split_dir / "labels", class_names, settings)


if __name__ == "__main__":
    main()
