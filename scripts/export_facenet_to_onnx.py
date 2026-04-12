from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from facenet_pytorch import InceptionResnetV1


def export_model(output_path: Path, opset_version: int = 17) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch_home = Path(
        os.environ.get("TORCH_HOME", Path(tempfile.gettempdir()) / "torch_cache")
    )
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    torch_home.mkdir(parents=True, exist_ok=True)

    model = InceptionResnetV1(pretrained="vggface2").eval()
    sample = torch.randn(1, 3, 160, 160, dtype=torch.float32)

    torch.onnx.export(
        model,
        sample,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    return output_path


if __name__ == "__main__":
    export_model(Path("server/runtime/models/facenet_vggface2.onnx"))
