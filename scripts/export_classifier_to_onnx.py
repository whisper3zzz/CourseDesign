from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def export_classifier(
    weights_path: Path,
    label_map_path: Path,
    output_path: Path,
    opset_version: int = 17,
) -> Path:
    label_map = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(label_map))
    state_dict = torch.load(Path(weights_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    sample = torch.randn(1, 3, 160, 160, dtype=torch.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained classifier to ONNX.")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("server/runtime/classifier/classifier.pt"),
        help="Path to trained classifier weights.",
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=Path("server/runtime/classifier/label_map.json"),
        help="Path to classifier label map.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("server/runtime/classifier/classifier.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_classifier(
        weights_path=args.weights_path,
        label_map_path=args.label_map_path,
        output_path=args.output_path,
        opset_version=args.opset_version,
    )
