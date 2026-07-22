"""Run a small ToonOut-shaped tensor workload on an AMD ROCm device."""

from __future__ import annotations

import argparse
import time


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args(arguments)
    if args.size < 64 or args.iterations < 1:
        parser.error("size must be at least 64 and iterations must be positive")

    import torch
    import torch.nn.functional as functional

    if not getattr(torch.version, "hip", None):
        raise RuntimeError("This PyTorch build does not include ROCm support.")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm did not expose an AMD GPU through torch.cuda.")

    device = torch.device("cuda")
    convolution = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).eval().to(device)
    source = torch.rand(1, 3, args.size, args.size, device=device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.iterations):
            features = convolution(source)
            reduced = functional.interpolate(
                features,
                size=(args.size // 2, args.size // 2),
                mode="bilinear",
                align_corners=False,
            )
            mask = reduced.mean(dim=1, keepdim=True).sigmoid()
        result = mask.cpu()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if result.shape != (1, 1, args.size // 2, args.size // 2):
        raise RuntimeError(f"Unexpected output shape: {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise RuntimeError("ROCm smoke output contains non-finite values.")

    print(f"torch={torch.__version__}")
    print(f"rocm={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"input={tuple(source.shape)} output={tuple(result.shape)}")
    print(f"iterations={args.iterations} elapsed_seconds={elapsed:.3f}")
    print("AMD ROCm smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
