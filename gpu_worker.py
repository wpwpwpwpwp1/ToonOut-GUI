"""선택형 GPU 팩 안에서 실행되는 JSON-lines 추론 worker."""

import argparse
import json
import sys
import time
from pathlib import Path

from inference import ToonOutEngine
from performance import PerformanceMode, PerformancePolicy, apply_torch_policy


def emit(event: str, **payload) -> None:
    # 프로세스 간 JSON은 ASCII만 출력한다. json.loads가 \uXXXX를 다시
    # 유니코드로 복원하므로 Windows 코드페이지와 무관하게 한국어가 보존된다.
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=True),
        flush=True,
    )


def friendly_error(error: Exception) -> str:
    message = str(error).strip()
    lowered = message.lower()
    if "out of memory" in lowered:
        return "GPU 메모리가 부족합니다. 더 작은 배치로 다시 시도하세요."
    if "cuda" in lowered or "hip" in lowered or "driver" in lowered:
        return (
            "GPU를 시작하지 못했습니다. 그래픽 드라이버를 업데이트하고 "
            f"다시 시도하세요. 세부 정보: {message[:160]}"
        )
    return message[:240] or "GPU worker에서 알 수 없는 오류가 발생했습니다."


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--jobs-file", required=True)
    parser.add_argument("--cancel-file", required=True)
    parser.add_argument("--pause-file", required=True)
    parser.add_argument("--cpu-threads", required=True, type=int)
    parser.add_argument("--gpu-memory-fraction", required=True, type=float)
    parser.add_argument("--cooldown-ms", required=True, type=int)
    args = parser.parse_args(arguments)

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU compute 장치를 사용할 수 없습니다. 그래픽 드라이버와 "
                "가속 팩 호환성을 확인하세요."
            )
        apply_torch_policy(
            torch,
            PerformancePolicy(
                PerformanceMode.CUSTOM,
                args.cpu_threads,
                args.gpu_memory_fraction,
                args.cooldown_ms,
            ),
        )
        engine = ToonOutEngine(args.model_directory)
        engine.load(lambda message: emit("model_status", message=message))
        if engine.device_label == "CPU":
            raise RuntimeError("GPU worker가 CPU 장치를 선택했습니다.")
        emit("model_ready", device_label=engine.device_label)
    except Exception as error:
        emit("model_failed", error=friendly_error(error))
        return 2

    try:
        jobs = json.loads(Path(args.jobs_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        emit("model_failed", error=f"처리 목록을 읽지 못했습니다: {error}")
        return 2

    total = len(jobs)
    completed = 0
    cancelled = False
    cancel_file = Path(args.cancel_file)
    pause_file = Path(args.pause_file)
    last_completed_at: float | None = None

    def wait_before_next_item() -> bool:
        pause_announced = False
        while True:
            if cancel_file.exists():
                return False
            if pause_file.exists():
                if not pause_announced:
                    emit("batch_paused")
                    pause_announced = True
                time.sleep(0.1)
                continue
            if last_completed_at is None:
                return True
            elapsed_ms = (time.monotonic() - last_completed_at) * 1000
            remaining_ms = args.cooldown_ms - elapsed_ms
            if remaining_ms <= 0:
                return True
            time.sleep(min(0.1, remaining_ms / 1000))

    for item_id, source_path, output_path in jobs:
        if not wait_before_next_item():
            cancelled = True
            break
        emit("item_started", item_id=item_id)
        try:
            engine.remove_background(source_path, output_path)
            emit(
                "item_completed",
                item_id=item_id,
                output_path=output_path,
            )
        except Exception as error:
            emit(
                "item_failed",
                item_id=item_id,
                error=friendly_error(error),
            )
        completed += 1
        emit("progress_changed", completed=completed, total=total)
        last_completed_at = time.monotonic()

    emit("batch_finished", cancelled=cancelled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
