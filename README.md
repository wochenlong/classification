# classification

## Installation

```shell
pip install -r requirements.txt
pip install -r requirements-onnx.txt
```

## Let's Training the Classifier

Save this to `train.py`

```python
import math

from ditk import logging
from torchvision import transforms

from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset, RangeRandomCrop, prob_greyscale
from classification.train import train_simple

# 初始化日志记录
logging.try_init_root(logging.INFO)

# 任务元数据
LABELS = ['monochrome', 'normal']  # 每个类别的标签
WEIGHTS = [math.e ** 2, 1.0]  # 每个类别的权重
assert len(LABELS) == len(WEIGHTS), \
    f'标签和权重应该有相同的长度，但找到了{len(LABELS)}(标签)和{len(WEIGHTS)}(权重)。'

# 数据集目录（使用你自己的目录，如下所示）
# <dataset_dir>
# ├── class1
# │   ├── image1.jpg
# │   └── image2.png  # 所有PIL可读格式都可以
# ├── class2
# │   ├── image3.jpg
# │   └── image4.jpg
# └── class3（注意：原代码中未使用class3，可能是示例的一部分）
#     ├── image5.jpg
#     └── image6.jpg
DATASET_DIR = '/my/dataset/directory'

# 训练数据集的数据增强和预处理
TRANSFORM_TRAIN = transforms.Compose([
    # 数据增强
    # prob_greyscale(0.5),  # 当颜色不重要时使用这行
    transforms.Resize((500, 500)),
    RangeRandomCrop((400, 500), padding=0, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-45, 45)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.05, 0.03),

    # 预处理（建议与测试时相同）
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 测试数据集的预处理
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
# 可视化数据集（当类别太多时可能会很慢）
# 如果不需要，就注释掉
TRANSFORM_VISUAL = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# 准备数据集
# 在大型数据集上训练时禁用缓存
dataset = LocalImageDataset(DATASET_DIR, LABELS, no_cache=True)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])
train_dataset = WrappedImageDataset(train_dataset, TRANSFORM_TRAIN)
# 如果不需要可视化，就使用这行
# test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST)
# 否则，使用带有可视化变换的测试数据集
test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST, TRANSFORM_VISUAL)

# 开始训练！
if __name__ == '__main__':
    train_simple(
        # 训练任务的工作目录
        # 中断后会自动恢复
        workdir='runs/demo_exp',

        # 所有timm中的模型都可使用，
        # 查看支持的模型请使用timm.list_models()或查看性能表
        # https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
        # 推荐：
        # 1. 使用caformer_s36.sail_in22k_ft_in1k_384进行训练
        # 2. 使用mobilenetv3_large_100进行蒸馏
        model_name='caformer_s36.sail_in22k_ft_in1k_384',

        # 标签和权重，未给出权重时全部为1
        labels=LABELS,
        loss_weight=WEIGHTS,

        # 数据集
        train_dataset=train_dataset,
        test_dataset=test_dataset,

        # 训练设置，将使用预训练模型
        max_epochs=100,
        num_workers=8,
        eval_epoch=1,
        key_metric='accuracy',
        loss='focal',  # 当数据集不能保证干净时使用`sce`
        seed=0,
        drop_path_rate=0.4,  # 训练caformer时使用

        # 超参数
        batch_size=16,
        learning_rate=1e-5,  # caformer微调时推荐使用1e-5
        weight_decay=1e-3,
    )

```

And run

```
accelerate launch train.py
```

## Distillate the Model

Save this to `dist.py`

```python
from ditk import logging
from torchvision import transforms

from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset, RangeRandomCrop, prob_greyscale
from classification.train import train_distillation

# 初始化日志记录器
logging.try_init_root(logging.INFO)

# 任务元数据
LABELS = ['monochrome', 'normal']  # 每个类别的标签
WEIGHTS = [1.0, 1.0]  # 每个类别的权重（此处设为相同）
assert len(LABELS) == len(WEIGHTS), \
    f'标签和权重应该有相同的长度，但找到了{len(LABELS)}(标签)和{len(WEIGHTS)}(权重)。'

# 数据集目录（使用你自己的目录）
# <dataset_dir>
# ├── class1
# │   ├── image1.jpg
# │   └── image2.png  # 所有PIL可读格式均可
# ├── class2
# │   ├── image3.jpg
# │   └── image4.jpg
# └── class3（注意：此脚本中未使用class3，可能是示例的一部分）
#     ├── image5.jpg
#     └── image6.jpg
DATASET_DIR = '/my/dataset/directory'

# 训练数据集的数据增强和预处理
TRANSFORM_TRAIN = transforms.Compose([
    # 数据增强
    # prob_greyscale(0.5),  # 当颜色信息不重要时使用
    transforms.Resize((500, 500)),
    RangeRandomCrop((400, 500), padding=0, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-45, 45)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.05, 0.03),

    # 预处理（建议与测试时相同）
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 测试数据集的预处理
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
# 可视化数据集（类别太多时可能会很慢）
# 如果不需要，就注释掉
TRANSFORM_VISUAL = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# 准备数据集
# 在大型数据集上训练时禁用缓存
dataset = LocalImageDataset(DATASET_DIR, LABELS, no_cache=True)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])
train_dataset = WrappedImageDataset(train_dataset, TRANSFORM_TRAIN)
test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST, TRANSFORM_VISUAL)
# 如果不需要可视化，就使用这行
# test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST)

# 开始蒸馏训练！
if __name__ == '__main__':
    train_distillation(
        # 蒸馏任务中学生模型的工作目录
        # 中断后会自动恢复
        workdir='runs/demo_exp_dist',

        # timm中的所有模型都可用
        # 使用timm.list_models()查看支持的模型，或查看性能表
        # https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
        # 推荐：
        # 1. 使用caformer_s36.sail_in22k_ft_in1k_384进行训练
        # 2. 使用mobilenetv3_large_100进行蒸馏
        model_name='mobilenetv3_large_100',

        # 从runs/demo_exp中的教师模型进行蒸馏
        teacher_workdir='runs/demo_exp',

        # 标签和权重，未给出权重时全部为1
        labels=LABELS,
        loss_weight=WEIGHTS,

        # 数据集
        train_dataset=train_dataset,
        test_dataset=test_dataset,

        # 训练设置，将使用预训练模型
        max_epochs=500,
        num_workers=8,
        eval_epoch=5,
        key_metric='accuracy',
        cls_loss='focal',  # 当数据集不能保证干净时使用`sce`
        seed=0,
        # drop_path_rate=0.4,  # 训练caformer时使用

        # 蒸馏设置
        temperature=7.0,
        alpha=0.3,

        # 超参数
        batch_size=16,
        learning_rate=1e-4,  # 蒸馏时推荐使用的学习率，caformer微调时推荐使用1e-5
        weight_decay=1e-3,
    )

```

And then run

```shell
accelerate launch dist.py
```

## Export onnx model

### From checkpoint

Attention that this export is only supported for ckpt dumped with `classficiation` module, for it contains extra
information except state dict of the network.

```shell
python -m classification.onnx export --help
```

```text
Usage: python -m classification.onnx export [OPTIONS]

  Export existing checkpoint to onnx

Options:
  -i, --input FILE       Input checkpoint to export.  [required]
  -s, --imgsize INTEGER  Image size for input.  [default: 384]
  -D, --non-dynamic      Do not export model with dynamic input height and
                         width.
  -V, --verbose          Show verbose information.
  -o, --output FILE      Output file of onnx model
  -h, --help             Show this message and exit.
```

## From training work directory

```shell
python -m classification.onnx dump -w runs/demo_exp
```

Then the `runs/demo_exp/ckpts/best.ckpt` will be dumped to `runs/demo_exp/onnxs/best.onnx`

Here is the help information

```text
Usage: python -m classification.onnx dump [OPTIONS]

  Dump onnx model from existing work directory.

Options:
  -w, --workdir DIRECTORY  Work directory of the training.  [required]
  -s, --imgsize INTEGER    Image size for input.  [default: 384]
  -D, --non-dynamic        Do not export model with dynamic input height and
                           width.
  -V, --verbose            Show verbose information.
  -h, --help               Show this message and exit.
```

## Publish Trained Models

Before start, set the `HF_TOKEN` variable to your huggingface token

```shell
# on Linux
export HF_TOKEN=xxxxxxxx
```

Publish trained models (including ckpt, onnx, metrics data and figures) to huggingface
repository `your/huggingface_repo`

```shell
python -m classification.publish huggingface -w runs/your_model_dir -n name_of_the_model -r your/huggingface_repo
```

List all the models in given repository `your/huggingface_repo`, which can be used in README

```shell
python -m classification.list huggingface -r your/huggingface_repo
```

An example model repository: https://huggingface.co/deepghs/anime_style_ages
