"""UI와 독립적인 배치 상태 및 파일 이름 규칙."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_PIXELS = 100_000_000
MAX_OUTPUT_AFFIX_LENGTH = 48
MAX_OUTPUT_STEM_LENGTH = 220
INVALID_OUTPUT_NAME_CHARACTERS = '<>:"/\\|?*'


class ItemState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


STATE_PRESENTATION = {
    ItemState.QUEUED: ("○", "대기 중"),
    ItemState.PROCESSING: ("◌", "처리 중"),
    ItemState.COMPLETED: ("✓", "완료"),
    ItemState.FAILED: ("!", "실패"),
    ItemState.CANCELLED: ("—", "중지됨"),
}


@dataclass
class BatchItem:
    item_id: str
    source_path: str
    state: ItemState = ItemState.QUEUED
    result_path: str | None = None
    result_saved: bool = False
    error: str | None = None

    @property
    def filename(self) -> str:
        return Path(self.source_path).name


@dataclass(frozen=True)
class OutputNamingPolicy:
    prefix: str = ""
    suffix: str = ""

    @property
    def is_default(self) -> bool:
        return not self.prefix and not self.suffix

    def stem_for(self, source_path: str) -> str:
        source_stem = Path(source_path).stem
        available_source_length = max(
            1,
            MAX_OUTPUT_STEM_LENGTH - len(self.prefix) - len(self.suffix),
        )
        return (
            f"{self.prefix}{source_stem[:available_source_length]}"
            f"{self.suffix}"
        )

    def filename_for(self, source_path: str) -> str:
        return f"{self.stem_for(source_path)}.png"


def validate_output_affix(value: str) -> str | None:
    if len(value) > MAX_OUTPUT_AFFIX_LENGTH:
        return f"{MAX_OUTPUT_AFFIX_LENGTH}자 이하로 입력하세요"
    if any(ord(character) < 32 for character in value):
        return "줄바꿈이나 제어 문자는 사용할 수 없습니다"
    invalid = sorted(
        {
            character
            for character in value
            if character in INVALID_OUTPUT_NAME_CHARACTERS
        }
    )
    if invalid:
        return f"파일 이름에 사용할 수 없는 문자: {' '.join(invalid)}"
    if value.endswith((" ", ".")):
        return "끝에 공백이나 점을 사용할 수 없습니다"
    return None


def available_output_path(
    folder: Path,
    source_path: str,
    reserved_paths: set[Path] | None = None,
    *,
    naming_policy: OutputNamingPolicy | None = None,
) -> Path:
    reserved = {
        str(Path(path).resolve()).casefold()
        for path in (reserved_paths or set())
    }
    policy = naming_policy or OutputNamingPolicy()
    stem = policy.stem_for(source_path)
    candidate = folder / f"{stem}.png"

    def is_available(path: Path) -> bool:
        return (
            not path.exists()
            and str(path.resolve()).casefold() not in reserved
        )

    if is_available(candidate):
        return candidate

    if policy.is_default:
        collision_stem = f"{stem}_no_bg"
        candidate = folder / f"{collision_stem}.png"
        if is_available(candidate):
            return candidate
        suffix_number = 2
    else:
        collision_stem = stem
        suffix_number = 2

    while True:
        candidate = folder / f"{collision_stem}_{suffix_number}.png"
        if is_available(candidate):
            return candidate
        suffix_number += 1
