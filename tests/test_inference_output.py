import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from inference import (
    _combine_alpha,
    _existing_alpha,
    _publish_without_overwrite,
)


class ExistingAlphaTests(unittest.TestCase):
    def test_model_mask_is_limited_by_existing_transparency(self):
        source = Image.new("RGBA", (2, 1), (20, 30, 40, 255))
        source_alpha = Image.new("L", (2, 1), 0)
        source_alpha.putpixel((1, 0), 128)
        source.putalpha(source_alpha)
        model_mask = Image.new("L", (2, 1), 255)

        combined = _combine_alpha(model_mask, _existing_alpha(source))

        self.assertEqual(list(combined.get_flattened_data()), [0, 128])

    def test_opaque_rgb_source_keeps_the_model_mask(self):
        source = Image.new("RGB", (2, 1), (20, 30, 40))
        model_mask = Image.new("L", (2, 1))
        model_mask.putdata([32, 224])

        combined = _combine_alpha(model_mask, _existing_alpha(source))

        self.assertEqual(list(combined.get_flattened_data()), [32, 224])


class OutputPublishingTests(unittest.TestCase):
    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "result.tmp"
            destination = root / "result.png"
            temporary.write_bytes(b"new result")
            destination.write_bytes(b"existing result")

            with self.assertRaises(FileExistsError):
                _publish_without_overwrite(temporary, destination)

            self.assertEqual(destination.read_bytes(), b"existing result")
            self.assertEqual(temporary.read_bytes(), b"new result")

    def test_complete_temporary_file_is_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "result.tmp"
            destination = root / "result.png"
            temporary.write_bytes(b"complete result")

            _publish_without_overwrite(temporary, destination)

            self.assertEqual(destination.read_bytes(), b"complete result")
            self.assertFalse(temporary.exists())

    def test_filesystem_without_hard_links_uses_exclusive_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "result.tmp"
            destination = root / "result.png"
            temporary.write_bytes(b"fallback result")

            with patch("inference.os.link", side_effect=OSError("unsupported")):
                _publish_without_overwrite(temporary, destination)

            self.assertEqual(destination.read_bytes(), b"fallback result")
            self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
