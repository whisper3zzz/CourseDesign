from pathlib import Path

from scripts.export_facenet_to_onnx import export_model


def test_export_model_writes_onnx_file(tmp_path: Path) -> None:
    output_path = tmp_path / "facenet_vggface2.onnx"

    export_model(output_path=output_path, opset_version=17)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
