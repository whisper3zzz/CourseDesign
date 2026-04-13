from pathlib import Path

from server.classifier_assets import ClassifierArtifactPaths


def test_classifier_artifact_paths_use_classifier_directory(tmp_path: Path) -> None:
    paths = ClassifierArtifactPaths(root=tmp_path)

    assert paths.onnx_path == tmp_path / "classifier.onnx"
    assert paths.mindir_path == tmp_path / "classifier.mindir"
    assert paths.label_map_path == tmp_path / "label_map.json"
