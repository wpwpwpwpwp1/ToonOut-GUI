from pathlib import Path
import unittest

from PIL import Image

from scripts.create_app_icon import WINDOWS_ICON_SIZES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppIconTests(unittest.TestCase):
    def test_png_source_is_square_with_transparent_corners(self):
        with Image.open(PROJECT_ROOT / "assets" / "toonout-icon.png") as image:
            rgba = image.convert("RGBA")

        self.assertEqual(rgba.width, rgba.height)
        self.assertTrue(all(rgba.getpixel(point)[3] == 0 for point in (
            (0, 0),
            (rgba.width - 1, 0),
            (0, rgba.height - 1),
            (rgba.width - 1, rgba.height - 1),
        )))

    def test_windows_icon_contains_expected_resolutions(self):
        with Image.open(PROJECT_ROOT / "assets" / "toonout.ico") as icon:
            available_sizes = icon.info.get("sizes", set())

        expected_sizes = {(size, size) for size in WINDOWS_ICON_SIZES}
        self.assertTrue(expected_sizes.issubset(available_sizes))


if __name__ == "__main__":
    unittest.main()
