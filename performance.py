"""배치 처리 성능 정책과 PyTorch 런타임 적용 규칙."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


PERFORMANCE_MODE_SETTING = "performance/mode"
PERFORMANCE_CPU_THREADS_SETTING = "performance/cpu_threads"
PERFORMANCE_GPU_MEMORY_SETTING = "performance/gpu_memory_fraction"
PERFORMANCE_COOLDOWN_SETTING = "performance/cooldown_ms"


class PerformanceMode(str, Enum):
    ECO = "eco"
    BALANCED = "balanced"
    MAXIMUM = "maximum"
    CUSTOM = "custom"


MODE_LABELS = {
    PerformanceMode.ECO: "절약",
    PerformanceMode.BALANCED: "균형",
    PerformanceMode.MAXIMUM: "최대 성능",
    PerformanceMode.CUSTOM: "사용자 지정",
}


@dataclass(frozen=True)
class PerformancePolicy:
    mode: PerformanceMode
    cpu_threads: int
    gpu_memory_fraction: float
    cooldown_ms: int

    def normalized(self, logical_cpus: int | None = None) -> "PerformancePolicy":
        cpu_count = max(1, logical_cpus or os.cpu_count() or 1)
        return PerformancePolicy(
            mode=self.mode,
            cpu_threads=min(cpu_count, max(1, int(self.cpu_threads))),
            gpu_memory_fraction=min(
                1.0,
                max(0.1, float(self.gpu_memory_fraction)),
            ),
            cooldown_ms=min(10_000, max(0, int(self.cooldown_ms))),
        )


def preset_policy(
    mode: PerformanceMode,
    logical_cpus: int | None = None,
) -> PerformancePolicy:
    cpu_count = max(1, logical_cpus or os.cpu_count() or 1)
    if mode == PerformanceMode.ECO:
        return PerformancePolicy(
            mode,
            max(1, round(cpu_count * 0.25)),
            0.70,
            1500,
        )
    if mode == PerformanceMode.BALANCED:
        return PerformancePolicy(
            mode,
            max(1, round(cpu_count * 0.50)),
            0.85,
            500,
        )
    if mode == PerformanceMode.MAXIMUM:
        return PerformancePolicy(mode, cpu_count, 1.0, 0)
    raise ValueError("사용자 지정 모드에는 직접 지정한 정책이 필요합니다.")


def apply_torch_policy(torch_module, policy: PerformancePolicy) -> None:
    """모델을 만들기 전에 PyTorch가 사용할 CPU/VRAM 상한을 적용한다."""
    policy = policy.normalized()
    torch_module.set_num_threads(policy.cpu_threads)
    try:
        # 이 값은 프로세스에서 한 번만 바꿀 수 있으므로 재로드 시 오류는 무시한다.
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if (
        policy.gpu_memory_fraction < 1.0
        and torch_module.cuda.is_available()
    ):
        torch_module.cuda.set_per_process_memory_fraction(
            policy.gpu_memory_fraction
        )
