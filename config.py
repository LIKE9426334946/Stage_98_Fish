from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def load_config():
    """所有脚本共用项目目录中的 config.yaml。"""
    path = ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须包含 YAML 配置项: {path}")

    return config


def project_path(value):
    """项目配置中的相对路径统一以项目目录为基准。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def dataset_path(config):
    return project_path(config["dataset"]["root"])


def data_file(config):
    return dataset_path(config) / config["dataset"]["data_file"]


def model_name(config):
    name = config["model"]["name"]
    path = Path(name)
    # 保留 yolo11n.pt 这类模型名，允许 Ultralytics 自动下载预训练权重。
    if path.parent == Path(".") and not name.startswith("."):
        return name
    return str(project_path(name))


def weights_path(config):
    weights = config["model"]["weights"]
    if weights is not None:
        return project_path(weights)

    training = config["training"]
    return project_path(training["output_dir"]) / training["name"] / "weights" / "best.pt"


def select_device(config):
    import torch

    device = config["training"]["device"]
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device


def common_args(config):
    """训练、验证和测试使用相同的数据集、尺寸及设备配置。"""
    training = config["training"]
    return {
        "data": str(data_file(config)),
        "imgsz": config["dataset"]["image_size"],
        "batch": training["batch_size"],
        "device": select_device(config),
        "workers": training["num_workers"],
    }
