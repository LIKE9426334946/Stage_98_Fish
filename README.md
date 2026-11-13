# Stage_98_Fish

基于 Ultralytics YOLO 的鱼类目标检测项目。训练、验证、测试、预测及标注可视化均从项目目录下的 `config.yaml` 读取参数。

安装依赖：

```bash
pip install -r requirements.txt
```

先修改 `config.yaml` 中的 `dataset.root`，指向已有 YOLO 数据集目录。`dataset.data_file` 指定数据集自带的 `data.yaml`，该文件继续提供类别名称与训练、验证、测试集划分，不需要重新划分或预处理数据。

配置分组：

| 分组 | 配置内容 |
| --- | --- |
| `model` | YOLO 模型名称、评估和预测使用的权重 |
| `training` | 随机种子、轮数、批大小、学习率、优化器、设备、输出目录等 |
| `dataset` | 数据集根目录、数据描述文件、图像尺寸 |
| `evaluation` | 验证和测试的数据划分、结果目录和名称 |
| `prediction` | 预测图片路径、置信度、IoU 阈值、保存选项 |
| `visualization` | 标注可视化的数据目录、数量、随机种子及绘图样式 |

运行命令：

```bash
python train.py
python val.py
python test.py
python predict.py
python visualize.py
```

参数说明：

- `training.device: auto` 会在 CUDA 可用时使用 GPU 0，否则使用 CPU；也可设置为 `cpu` 或 GPU 编号。
- 保留原脚本的默认 YOLO11n 模型、100 轮训练、批大小 8、图像边长 640 和数据集路径。YOLO 训练和评估的 `image_size` 使用单个整数。
- `training.learning_rate` 对应 YOLO 的 `lr0`。保留的 `optimizer: auto` 会自动选择优化器和学习率；需要指定学习率时，将 `optimizer` 改为 `AdamW`、`SGD` 等。
- `model.weights: null` 时，验证、测试和预测自动读取 `training.output_dir/training.name/weights/best.pt`。修改训练目录或实验名后，权重路径会同步变化。也可填写已有权重的路径。
- 模型名称 `yolo11n.pt` 可由 Ultralytics 自动下载；本地模型文件也可填写绝对路径或相对于项目目录的路径。
- 数据集根目录、结果目录和显式权重路径中的相对路径，以项目目录为基准。`dataset.data_file` 和 `prediction.source` 的相对路径，以 `dataset.root` 为基准。
- `visualization.split` 填写数据集中的实际目录名，例如 `train`、`valid` 或 `test`；该目录下应有 `images` 和 `labels` 文件夹。
