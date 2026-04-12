from pathlib import Path

import onnx

from scripts.export_facenet_to_onnx import export_model


def test_export_model_writes_onnx_file(tmp_path: Path) -> None:
    output_path = tmp_path / "facenet_vggface2.onnx"

    export_model(output_path=output_path, opset_version=17)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    assert [input.name for input in model.graph.input] == ["input"]
    assert [output.name for output in model.graph.output] == ["embedding"]
