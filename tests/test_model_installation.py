import tempfile
import unittest
from pathlib import Path

from inference import (
    BASE_MODEL_REPOSITORY,
    BASE_MODEL_REVISION,
    TOONOUT_REPOSITORY,
    TOONOUT_REVISION,
    TOONOUT_WEIGHTS,
)
from model_installation import (
    BASE_MODEL_CODE_FILES,
    delete_model_files,
    model_is_installed,
    model_storage_size,
    move_model_files,
    repository_directory,
    snapshot_directory,
)


def create_fake_installation(directory: Path):
    base_snapshot = snapshot_directory(
        directory,
        BASE_MODEL_REPOSITORY,
        BASE_MODEL_REVISION,
    )
    base_snapshot.mkdir(parents=True)
    for filename in BASE_MODEL_CODE_FILES:
        (base_snapshot / filename).write_text("# fixture", encoding="utf-8")

    toonout_snapshot = snapshot_directory(
        directory,
        TOONOUT_REPOSITORY,
        TOONOUT_REVISION,
    )
    toonout_snapshot.mkdir(parents=True)
    (toonout_snapshot / TOONOUT_WEIGHTS).write_bytes(b"toonout-weights")


class ModelInstallationStatusTests(unittest.TestCase):
    def test_complete_cache_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_fake_installation(root)

            self.assertTrue(model_is_installed(root))
            self.assertGreater(model_storage_size(root), 0)
            self.assertFalse(
                (snapshot_directory(
                    root,
                    BASE_MODEL_REPOSITORY,
                    BASE_MODEL_REVISION,
                ) / "model.safetensors").exists()
            )

    def test_partial_cache_is_not_reported_as_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_snapshot = snapshot_directory(
                root,
                BASE_MODEL_REPOSITORY,
                BASE_MODEL_REVISION,
            )
            base_snapshot.mkdir(parents=True)
            (base_snapshot / "config.json").write_text("{}", encoding="utf-8")

            self.assertFalse(model_is_installed(root))


class ModelFileOperationTests(unittest.TestCase):
    def test_move_verifies_destination_then_removes_only_source_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "keep.txt").write_text("keep", encoding="utf-8")
            create_fake_installation(source)

            source_removed = move_model_files(source, destination)

            self.assertTrue(source_removed)
            self.assertTrue(model_is_installed(destination))
            self.assertFalse(model_is_installed(source))
            self.assertTrue((source / "keep.txt").is_file())

    def test_delete_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_fake_installation(root)
            (root / "keep.txt").write_text("keep", encoding="utf-8")

            delete_model_files(root)

            self.assertFalse(model_is_installed(root))
            self.assertTrue((root / "keep.txt").is_file())
            for repository in (BASE_MODEL_REPOSITORY, TOONOUT_REPOSITORY):
                self.assertFalse(repository_directory(root, repository).exists())

    def test_move_rejects_destination_inside_current_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            create_fake_installation(source)

            with self.assertRaises(ValueError):
                move_model_files(source, source / "nested")


if __name__ == "__main__":
    unittest.main()
