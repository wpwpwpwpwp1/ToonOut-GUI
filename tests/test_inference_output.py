import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from inference import (
    _combine_alpha,
    _existing_alpha,
    _publish_without_overwrite,
    cleanup_abandoned_output_files,
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
    def test_abandoned_temporary_output_is_removed_but_active_one_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abandoned = root / (
                ".result.png.toonout-2147483647-"
                "0123456789abcdef0123456789abcdef.tmp"
            )
            active = root / (
                f".result.png.toonout-{os.getpid()}-"
                "fedcba9876543210fedcba9876543210.tmp"
            )
            unrelated = root / "result.tmp"
            for path in (abandoned, active, unrelated):
                path.write_bytes(b"temporary")

            removed = cleanup_abandoned_output_files(root)

            self.assertEqual(removed, 1)
            self.assertFalse(abandoned.exists())
            self.assertTrue(active.exists())
            self.assertTrue(unrelated.exists())

    def test_old_legacy_temporary_output_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / (
                ".result.png.0123456789abcdef0123456789abcdef.tmp"
            )
            legacy.write_bytes(b"old temporary")
            old = time.time() - 2 * 24 * 60 * 60
            os.utime(legacy, (old, old))

            cleanup_abandoned_output_files(root)

            self.assertFalse(legacy.exists())

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
