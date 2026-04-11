from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, nn
from mindspore.train import Callback, LossMonitor, Model, TimeMonitor
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mindface.recognition.models.mobilefacenet import get_mbf


class MobileFaceNetClassifier(nn.Cell):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 512,
        backbone: nn.Cell | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else get_mbf(num_features=feature_dim)
        self.classifier = nn.Dense(feature_dim, num_classes)

    def construct(self, x: Tensor) -> Tensor:
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return logits


class EvalCallback(Callback):
    def __init__(self, model: Model, val_dataset, state: dict[str, object]) -> None:
        super().__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.state = state

    def on_train_epoch_end(self, run_context) -> None:
        result = self.model.eval(self.val_dataset, dataset_sink_mode=False)
        self.state["last_val_accuracy"] = float(result["acc"])
        print(f"val_acc: {result['acc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local MobileFaceNet classifier on a prepared face subset."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Prepared training directory in ImageFolder layout.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        required=True,
        help="Prepared validation directory in ImageFolder layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints and training metadata.",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("utils/mindface/recognition/pretrained/mobile_casia_ArcFace.ckpt"),
        help="Optional pretrained MobileFaceNet checkpoint.",
    )
    parser.add_argument(
        "--device-target",
        choices=("CPU",),
        default="CPU",
        help="MindSpore device target. This script is intended for local CPU training.",
    )
    parser.add_argument(
        "--running-mode",
        choices=("PYNATIVE", "GRAPH"),
        default="GRAPH",
        help="MindSpore execution mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=512,
        help="Embedding dimension produced by MobileFaceNet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataset worker count.",
    )
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Fine-tune the backbone instead of only training the classifier head.",
    )
    return parser.parse_args()


def build_dataset(
    root: Path,
    batch_size: int,
    num_workers: int,
    training: bool,
):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    dataset = ds.ImageFolderDataset(
        str(root),
        shuffle=training,
        num_parallel_workers=max(1, num_workers),
    )
    mean = [127.5, 127.5, 127.5]
    std = [127.5, 127.5, 127.5]
    image_ops = [
        vision.Decode(),
        vision.Resize((112, 112)),
    ]
    if training:
        image_ops.append(vision.RandomHorizontalFlip(prob=0.5))
    image_ops.extend(
        [
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW(),
        ]
    )
    dataset = dataset.map(
        operations=image_ops,
        input_columns="image",
        num_parallel_workers=max(1, num_workers),
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(ms.int32),
        input_columns="label",
        num_parallel_workers=max(1, num_workers),
    )
    dataset = dataset.batch(batch_size, drop_remainder=training)
    return dataset


def collect_label_names(root: Path) -> list[str]:
    labels = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    if not labels:
        raise RuntimeError(f"No class folders found under {root}")
    return labels


def configure_context(device_target: str, running_mode: str) -> None:
    mode = ms.PYNATIVE_MODE if running_mode == "PYNATIVE" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target=device_target)


def verify_conv_runtime(device_target: str) -> None:
    try:
        probe = nn.Conv2d(
            3,
            8,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
        )
        sample = Tensor(np.random.rand(1, 3, 16, 16).astype(np.float32))
        _ = probe(sample)
    except Exception as exc:
        raise RuntimeError(
            "MindSpore Conv2d preflight failed on the current runtime. "
            "This usually means the local macOS CPU backend can import MindSpore "
            "but still cannot execute CNN operators. "
            f"Current device_target={device_target}. "
            "Use a Linux/GPU/Ascend environment, or switch the CNN training path "
            "to another framework on this machine."
        ) from exc


def main() -> None:
    args = parse_args()
    ms.set_seed(args.seed)
    configure_context(args.device_target, args.running_mode)
    verify_conv_runtime(args.device_target)

    label_names = collect_label_names(args.train_dir)
    if collect_label_names(args.val_dir) != label_names:
        raise RuntimeError("Training and validation labels do not match.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(
        root=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training=True,
    )
    val_dataset = build_dataset(
        root=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training=False,
    )

    train_steps = train_dataset.get_dataset_size()
    val_steps = val_dataset.get_dataset_size()
    if train_steps <= 0:
        raise RuntimeError("Training dataset is empty.")
    if val_steps <= 0:
        raise RuntimeError(
            "Validation dataset is empty. Try lowering --batch-size or rebuilding the subset."
        )

    backbone = get_mbf(num_features=args.feature_dim)
    if args.pretrained and args.pretrained.is_file():
        params = load_checkpoint(str(args.pretrained))
        load_param_into_net(backbone, params, strict_load=False)

    network = MobileFaceNetClassifier(
        num_classes=len(label_names),
        feature_dim=args.feature_dim,
        backbone=backbone,
    )

    if not args.train_backbone:
        for parameter in network.backbone.trainable_params():
            parameter.requires_grad = False
        trainable_params = network.classifier.trainable_params()
    else:
        trainable_params = network.trainable_params()

    optimizer = nn.Adam(trainable_params, learning_rate=args.learning_rate)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    model = Model(
        network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={"acc": nn.Accuracy()},
    )

    checkpoint_config = CheckpointConfig(
        save_checkpoint_steps=train_steps,
        keep_checkpoint_max=5,
    )
    callbacks = [
        TimeMonitor(data_size=train_steps),
        LossMonitor(),
        ModelCheckpoint(
            prefix="mobilefacenet_subset",
            directory=str(checkpoint_dir),
            config=checkpoint_config,
        ),
    ]
    summary: dict[str, object] = {
        "config": {
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
            "output_dir": str(args.output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "feature_dim": args.feature_dim,
            "seed": args.seed,
            "train_backbone": args.train_backbone,
            "device_target": args.device_target,
            "running_mode": args.running_mode,
        },
        "labels": label_names,
    }
    callbacks.append(EvalCallback(model=model, val_dataset=val_dataset, state=summary))

    model.train(
        args.epochs,
        train_dataset,
        callbacks=callbacks,
        dataset_sink_mode=False,
    )

    final_metrics = model.eval(val_dataset, dataset_sink_mode=False)
    summary["final_val_accuracy"] = float(final_metrics["acc"])

    backbone_path = args.output_dir / "mobilefacenet_backbone.ckpt"
    classifier_path = args.output_dir / "mobilefacenet_classifier.ckpt"
    label_map_path = args.output_dir / "label_map.json"
    summary_path = args.output_dir / "train_summary.json"

    ms.save_checkpoint(network.backbone, str(backbone_path))
    ms.save_checkpoint(network, str(classifier_path))
    label_map_path.write_text(
        json.dumps(
            {
                "num_classes": len(label_names),
                "labels": label_names,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Training completed.")
    print(f"train_steps: {train_steps}")
    print(f"val_steps: {val_steps}")
    print(f"final_val_accuracy: {final_metrics['acc']:.4f}")
    print(f"backbone_ckpt: {backbone_path}")
    print(f"classifier_ckpt: {classifier_path}")
    print(f"label_map: {label_map_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
