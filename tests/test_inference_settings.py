import unittest
from pathlib import Path

import torch

from inference import (
    ToonOutEngine,
    _download_progress_class,
    prepare_model_for_device,
    verify_model_runtime_dependencies,
)


class InferenceModelDirectoryTests(unittest.TestCase):
    def test_download_progress_maps_bytes_to_install_percent_range(self):
        events = []
        progress_class = _download_progress_class(
            lambda status, percent: events.append((status, percent)),
            "가중치 다운로드 중",
            25,
            85,
        )
        progress = progress_class(total=100)

        progress.update(50)
        progress.update(50)
        progress.close()

        self.assertEqual(
            events,
            [("가중치 다운로드 중", 55), ("가중치 다운로드 중", 85)],
        )

    def test_required_remote_code_dependencies_are_available(self):
        verify_model_runtime_dependencies()

    def test_model_directory_can_change_before_loading(self):
        engine = ToonOutEngine("C:/first-model-folder")

        changed = engine.set_model_directory("D:/second-model-folder")

        self.assertTrue(changed)
        self.assertEqual(engine.model_directory, Path("D:/second-model-folder"))

    def test_cpu_model_and_input_use_float32(self):
        half_model = torch.nn.Conv2d(3, 4, kernel_size=1).half()

        model, input_dtype = prepare_model_for_device(half_model, "cpu")

        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertEqual(input_dtype, torch.float32)

    def test_loaded_model_keeps_its_current_directory(self):
        engine = ToonOutEngine("C:/first-model-folder")
        engine._model = object()

        changed = engine.set_model_directory("D:/second-model-folder")

        self.assertFalse(changed)
        self.assertEqual(engine.model_directory, Path("C:/first-model-folder"))


if __name__ == "__main__":
    unittest.main()
