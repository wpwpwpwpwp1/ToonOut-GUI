import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app_settings import (
    available_storage_bytes,
    configure_huggingface_environment,
    default_model_directory,
    default_update_directory,
    ensure_writable_model_directory,
    format_storage_size,
)


class ModelDirectoryTests(unittest.TestCase):
    def test_default_location_uses_local_app_data(self):
        result = default_model_directory("C:/Users/test/AppData/Local")

        self.assertEqual(
            result,
            Path("C:/Users/test/AppData/Local/ToonOut/models"),
        )

    def test_writable_check_creates_directory_and_removes_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory) / "ToonOut" / "models"

            result = ensure_writable_model_directory(model_directory)

            self.assertEqual(result, model_directory)
            self.assertTrue(model_directory.is_dir())
            self.assertEqual(list(model_directory.iterdir()), [])

    def test_update_location_uses_separate_local_app_data_folder(self):
        result = default_update_directory("C:/Users/test/AppData/Local")

        self.assertEqual(
            result,
            Path("C:/Users/test/AppData/Local/ToonOut/updates"),
        )

    def test_available_space_accepts_a_directory_not_created_yet(self):
        with tempfile.TemporaryDirectory() as directory:
            result = available_storage_bytes(Path(directory) / "future" / "models")

        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_huggingface_cache_environment_uses_selected_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_huggingface_environment("D:/ToonOut/models")

            self.assertEqual(
                Path(os.environ["HF_HOME"]),
                Path("D:/ToonOut/models"),
            )
            self.assertEqual(
                Path(os.environ["HF_MODULES_CACHE"]),
                Path("D:/ToonOut/models/modules"),
            )


class StorageSizeTests(unittest.TestCase):
    def test_formats_gigabytes_for_the_dialog(self):
        self.assertEqual(format_storage_size(3 * 1024**3), "3.0 GB")


if __name__ == "__main__":
    unittest.main()
