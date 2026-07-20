import tempfile
import unittest
from pathlib import Path

from app_core import (
    BatchItem,
    ItemState,
    OutputNamingPolicy,
    available_output_path,
    validate_output_affix,
)


class BatchItemTests(unittest.TestCase):
    def test_filename_uses_only_the_final_path_component(self):
        item = BatchItem(
            item_id="item-1",
            source_path="C:/images/character.png",
        )

        self.assertEqual(item.filename, "character.png")
        self.assertEqual(item.state, ItemState.QUEUED)


class OutputPathTests(unittest.TestCase):
    def test_uses_original_name_when_it_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            result = available_output_path(
                Path(directory),
                "C:/images/character.jpg",
            )

        self.assertEqual(result.name, "character.png")

    def test_uses_no_bg_suffix_when_original_name_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "character.png").touch()

            result = available_output_path(
                folder,
                "C:/images/character.jpg",
            )

        self.assertEqual(result.name, "character_no_bg.png")

    def test_never_overwrites_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "character.png").touch()
            (folder / "character_no_bg.png").touch()
            (folder / "character_no_bg_2.png").touch()

            result = available_output_path(
                folder,
                "C:/images/character.jpg",
            )

        self.assertEqual(result.name, "character_no_bg_3.png")

    def test_reserved_batch_name_is_not_reused_before_file_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reserved = {folder / "character.png"}

            result = available_output_path(
                folder,
                "C:/another-folder/character.webp",
                reserved,
            )

        self.assertEqual(result.name, "character_no_bg.png")

    def test_custom_prefix_and_suffix_are_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            result = available_output_path(
                Path(directory),
                "C:/images/character.webp",
                naming_policy=OutputNamingPolicy("toonout_", "_cutout"),
            )

        self.assertEqual(result.name, "toonout_character_cutout.png")

    def test_custom_name_collision_uses_a_number(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "toonout_character_cutout.png").touch()
            result = available_output_path(
                folder,
                "C:/images/character.webp",
                naming_policy=OutputNamingPolicy("toonout_", "_cutout"),
            )

        self.assertEqual(result.name, "toonout_character_cutout_2.png")

    def test_output_affixes_reject_unsafe_windows_characters(self):
        self.assertIsNotNone(validate_output_affix("bad:name"))
        self.assertIsNotNone(validate_output_affix("trailing."))
        self.assertIsNone(validate_output_affix("toonout_"))

    def test_long_original_stem_is_shortened_but_affixes_are_kept(self):
        policy = OutputNamingPolicy("toonout_", "_cutout")

        result = policy.filename_for(f"{'a' * 250}.jpg")

        self.assertTrue(result.startswith("toonout_"))
        self.assertTrue(result.endswith("_cutout.png"))
        self.assertLessEqual(len(Path(result).stem), 220)


if __name__ == "__main__":
    unittest.main()
