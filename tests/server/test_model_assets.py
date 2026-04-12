from pathlib import Path

from server.model_assets import ModelArtifactPaths


def test_artifact_paths_point_into_runtime_models(tmp_path: Path) -> None:
    paths = ModelArtifactPaths(root=tmp_path)

    assert paths.onnx_path == tmp_path / "facenet_vggface2.onnx"
    assert paths.mindir_path == tmp_path / "facenet_vggface2.mindir"
    assert paths.meta_path == tmp_path / "facenet_vggface2.meta.json"
