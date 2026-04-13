import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from scripts.export_classifier_to_onnx import export_classifier


def test_export_classifier_writes_onnx_file(tmp_path: Path) -> None:
    weights_path = tmp_path / "classifier.pt"
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    torch.save(model.state_dict(), weights_path)
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(
        json.dumps({"0": "alice"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "classifier.onnx"

    export_classifier(
        weights_path=weights_path,
        label_map_path=label_map_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
