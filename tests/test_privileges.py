import sys
import unittest

from privileges import elevated_launch_arguments


class ElevatedLaunchTests(unittest.TestCase):
    def test_restart_preserves_model_location_and_selected_images(self):
        executable, parameters, _ = elevated_launch_arguments(
            "C:/Protected ToonOut/models",
            ["C:/Images/one image.png", "C:/Images/two.png"],
        )

        self.assertEqual(executable, sys.executable)
        self.assertIn("--model-directory", parameters)
        self.assertIn('"C:/Protected ToonOut/models"', parameters)
        self.assertEqual(parameters.count("--restore-image"), 2)


if __name__ == "__main__":
    unittest.main()
