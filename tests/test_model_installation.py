import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference import (
    BASE_MODEL_REPOSITORY,
    BASE_MODEL_REVISION,
    TOONOUT_REPOSITORY,
    TOONOUT_REVISION,
    TOONOUT_WEIGHTS,
)
from model_installation import (
    BASE_MODEL_CODE_FILES,
    cleanup_model_transfer_artifacts,
    delete_model_files,
    model_is_installed,
    model_storage_size,
    move_model_files,
    prepare_model_cache_for_install,
    repository_directory,
    snapshot_directory,
)
from model_files import has_link_free_layout, mark_link_free_layout
from processing import run_model_cleanup_worker


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

    def test_untrusted_model_link_is_treated_as_not_installed(self):
        with patch.object(
            Path,
            "stat",
            side_effect=OSError(448, "untrusted mount point"),
        ):
            self.assertFalse(model_is_installed("C:/models"))

    def test_untrusted_model_link_is_removed_only_when_install_starts(self):
        messages = []
        with (
            patch.object(
                Path,
                "stat",
                side_effect=OSError(448, "untrusted mount point"),
            ),
            patch("model_installation.delete_model_files") as delete_model_files_mock,
        ):
            repaired = prepare_model_cache_for_install(
                "C:/models",
                messages.append,
            )

        self.assertTrue(repaired)
        delete_model_files_mock.assert_called_once_with("C:/models")
        self.assertIn("이전 모델 파일", messages[0])

    def test_partial_cache_is_removed_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial_repository = repository_directory(
                root,
                BASE_MODEL_REPOSITORY,
            )
            partial_repository.mkdir(parents=True)
            (partial_repository / "partial-download").write_bytes(b"partial")
            unrelated = root / "keep.txt"
            unrelated.write_text("keep", encoding="utf-8")

            repaired = prepare_model_cache_for_install(root)

            self.assertTrue(repaired)
            self.assertFalse(partial_repository.exists())
            self.assertTrue(unrelated.is_file())

    def test_link_free_partial_download_is_preserved_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial_snapshot = snapshot_directory(
                root,
                BASE_MODEL_REPOSITORY,
                BASE_MODEL_REVISION,
            )
            partial_snapshot.mkdir(parents=True)
            partial_file = partial_snapshot / "config.json"
            partial_file.write_text("{}", encoding="utf-8")
            mark_link_free_layout(root)
            messages = []

            repaired = prepare_model_cache_for_install(root, messages.append)

            self.assertFalse(repaired)
            self.assertTrue(partial_file.is_file())
            self.assertIn("이어받는 중", messages[0])


class ModelFileOperationTests(unittest.TestCase):
    def test_abandoned_model_transfer_is_removed_but_active_one_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abandoned = root / ".toonout-transfer-2147483647-dead"
            active = root / f".toonout-transfer-{os.getpid()}-active"
            abandoned.mkdir()
            active.mkdir()
            (abandoned / "partial.pth").write_bytes(b"unused")

            removed = cleanup_model_transfer_artifacts(root)

            self.assertEqual(removed, 1)
            self.assertFalse(abandoned.exists())
            self.assertTrue(active.exists())

    def test_cleanup_worker_removes_partial_models_but_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial_repository = repository_directory(
                root,
                TOONOUT_REPOSITORY,
            )
            partial_repository.mkdir(parents=True)
            (partial_repository / "partial-download").write_bytes(b"partial")
            unrelated = root / "keep.txt"
            unrelated.write_text("keep", encoding="utf-8")

            result = run_model_cleanup_worker(str(root))

            self.assertEqual(result, 0)
            self.assertFalse(partial_repository.exists())
            self.assertTrue(unrelated.is_file())

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
            self.assertTrue(has_link_free_layout(destination))
            self.assertFalse(model_is_installed(source))
            self.assertTrue((source / "keep.txt").is_file())

    def test_delete_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_fake_installation(root)
            (root / "keep.txt").write_text("keep", encoding="utf-8")
            local_module_cache = (
                root
                / "modules"
                / "transformers_modules"
                / BASE_MODEL_REVISION
            )
            local_module_cache.mkdir(parents=True)
            (local_module_cache / "birefnet.py").write_text(
                "# generated remote module",
                encoding="utf-8",
            )

            delete_model_files(root)

            self.assertFalse(model_is_installed(root))
            self.assertFalse(has_link_free_layout(root))
            self.assertTrue((root / "keep.txt").is_file())
            for repository in (BASE_MODEL_REPOSITORY, TOONOUT_REPOSITORY):
                self.assertFalse(repository_directory(root, repository).exists())
            self.assertFalse(local_module_cache.exists())

    def test_move_rejects_destination_inside_current_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            create_fake_installation(source)

            with self.assertRaises(ValueError):
                move_model_files(source, source / "nested")


if __name__ == "__main__":
    unittest.main()
