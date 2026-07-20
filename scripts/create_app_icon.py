from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "assets" / "toonout-icon.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "toonout.ico"
WINDOWS_ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def create_windows_icon(source: Path, output: Path) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGBA")

    if image.width != image.height:
        raise ValueError("The app icon source must be square.")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output,
        format="ICO",
        sizes=[(size, size) for size in WINDOWS_ICON_SIZES],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the multi-resolution Windows icon used by ToonOut."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    create_windows_icon(arguments.source, arguments.output)
    print(f"Windows icon created: {arguments.output}")


if __name__ == "__main__":
    main()
