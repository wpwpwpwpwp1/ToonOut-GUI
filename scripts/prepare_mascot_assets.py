from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIRECTORY = PROJECT_ROOT / "tsunao"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "assets" / "mascot"
SOURCE_TO_OUTPUT = {
    "_tsunao.png": "icon.png",
    "_hit.png": "hit.png",
    "_tsunao.sleeping1.png": "asleep.png",
    "_tsunao.sleeping2.png": "asleep-2.png",
    "_tsunao.awake1.png": "awake-1.png",
    "_tsunao.awake2.png": "awake-2.png",
    "_tsunao.standing1.png": "standing-1.png",
    "_tsunao.standing2.png": "standing-2.png",
    "_tsunao.walking1.png": "walking-1.png",
    "_tsunao.walking2.png": "walking-2.png",
    "_tsunao.smug1.png": "smug-1.png",
    "_tsunao.smug2.png": "smug-2.png",
    "_tsunao.touchear1.png": "touchear-1.png",
    "_tsunao.touchear2.png": "touchear-2.png",
    "_tsunao.drowsy1.png": "drowsy-1.png",
    "_tsunao.drowsy2.png": "drowsy-2.png",
    "_tsunao.drawing1.png": "drawing-1.png",
    "_tsunao.drawing2.png": "drawing-2.png",
    "_tsunao.complete1.png": "complete.png",
    "_tsunao.complete2.png": "complete-2.png",
    "_tsunao.picked1.png": "picked-1.png",
    "_tsunao.picked2.png": "picked-2.png",
    "_tsunao.angry1.png": "angry-1.png",
    "_tsunao.angry2.png": "angry-2.png",
    "_tsunao.flying1.png": "flying-1.png",
    "_tsunao.flying2.png": "flying-2.png",
    "_tsunao.flying3.png": "flying-3.png",
    "_tsunao.flying4.png": "flying-4.png",
    "_tsunao.hurt1.png": "hurt-1.png",
    "_tsunao.hurt2.png": "hurt-2.png",
}
OUTPUT_SIZE = 640
STATE_CONTENT_EXTENT = 552
BOTTOM_ALIGNED_FRAMES = {
    "_tsunao.standing2.png",
    "_tsunao.angry2.png",
}


def normalize_state_frame(
    image: Image.Image,
    *,
    group_extent: int,
    bottom_align: bool = False,
) -> Image.Image:
    """Use one scale per animation group so facial size stays consistent."""
    alpha_bounds = image.getchannel("A").getbbox()
    if alpha_bounds is None:
        raise ValueError("Mascot frame has no visible pixels")
    content = image.crop(alpha_bounds)
    scale = STATE_CONTENT_EXTENT / group_extent
    target_size = (
        max(1, round(content.width * scale)),
        max(1, round(content.height * scale)),
    )
    content = content.resize(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGBA",
        (OUTPUT_SIZE, OUTPUT_SIZE),
        (0, 0, 0, 0),
    )
    x = (OUTPUT_SIZE - content.width) // 2
    y = (OUTPUT_SIZE - content.height) // 2
    if bottom_align:
        y = round(OUTPUT_SIZE / 2 + STATE_CONTENT_EXTENT / 2 - content.height)
    canvas.alpha_composite(content, (x, y))
    return canvas


def _state_group(source_name: str) -> str:
    return Path(source_name).stem.rstrip("0123456789")


def prepare_mascot_assets(source_directory: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    source_images: dict[str, Image.Image] = {}
    for source_name in SOURCE_TO_OUTPUT:
        source = source_directory / source_name
        if not source.is_file():
            raise FileNotFoundError(f"Mascot source was not found: {source}")

        with Image.open(source) as opened:
            source_images[source_name] = ImageOps.exif_transpose(opened).convert(
                "RGBA"
            )

    state_source_names = [
        source_name
        for source_name in SOURCE_TO_OUTPUT
        if source_name.startswith("_tsunao.")
        and source_name != "_tsunao.png"
    ]
    group_extents: dict[str, int] = {}
    for source_name in state_source_names:
        bounds = source_images[source_name].getchannel("A").getbbox()
        if bounds is None:
            raise ValueError(f"Mascot frame has no visible pixels: {source_name}")
        extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        group = _state_group(source_name)
        group_extents[group] = max(group_extents.get(group, 0), extent)

    for source_name, output_name in SOURCE_TO_OUTPUT.items():
        source = source_directory / source_name
        image = source_images[source_name]
        if source_name == "_hit.png":
            image.thumbnail(
                (OUTPUT_SIZE, OUTPUT_SIZE),
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new(
                "RGBA",
                (OUTPUT_SIZE, OUTPUT_SIZE),
                (0, 0, 0, 0),
            )
            canvas.alpha_composite(
                image,
                (
                    (OUTPUT_SIZE - image.width) // 2,
                    OUTPUT_SIZE - image.height,
                ),
            )
            image = canvas
        elif source_name in state_source_names:
            if image.width != image.height:
                raise ValueError(f"Mascot source must be square: {source}")
            image = normalize_state_frame(
                image,
                group_extent=group_extents[_state_group(source_name)],
                bottom_align=source_name in BOTTOM_ALIGNED_FRAMES,
            )
        else:
            if image.width != image.height:
                raise ValueError(f"Mascot source must be square: {source}")
            image = image.resize(
                (OUTPUT_SIZE, OUTPUT_SIZE),
                Image.Resampling.LANCZOS,
            )
        image.save(output_directory / output_name, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Tsunao mascot frames for the desktop app bundle."
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()

    prepare_mascot_assets(
        arguments.source_directory,
        arguments.output_directory,
    )
    print(f"Mascot assets created: {arguments.output_directory}")


if __name__ == "__main__":
    main()
