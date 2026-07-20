"""ToonOut 모델을 UI 코드에서 분리한 추론 어댑터."""

import os
import warnings
from pathlib import Path
from typing import Callable
from uuid import uuid4

from app_settings import configure_huggingface_environment, default_model_directory


BASE_MODEL_REPOSITORY = "ZhengPeng7/BiRefNet"
BASE_MODEL_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
TOONOUT_REPOSITORY = "joelseytre/toonout"
TOONOUT_REVISION = "cbf720eca394edcde66b861a8a8c20fbabe9c748"
TOONOUT_WEIGHTS = "birefnet_finetuned_toonout.pth"


def _existing_alpha(image):
    """원본에 투명도가 있으면 파일과 분리된 알파 채널을 반환한다."""
    if "A" not in image.getbands() and "transparency" not in image.info:
        return None
    return image.convert("RGBA").getchannel("A").copy()


def _combine_alpha(model_mask, source_alpha):
    """원본보다 불투명해지는 픽셀이 없도록 두 알파를 결합한다."""
    from PIL import Image, ImageChops

    mask = model_mask.convert("L")
    if source_alpha is None:
        return mask
    if source_alpha.size != mask.size:
        source_alpha = source_alpha.resize(
            mask.size,
            Image.Resampling.LANCZOS,
        )
    return ImageChops.multiply(mask, source_alpha.convert("L"))


def _publish_without_overwrite(temporary: Path, destination: Path) -> None:
    """완성된 임시 파일을 게시하되 기존 이름은 원자적으로 거부한다."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise
    except OSError:
        # 일부 이동식·네트워크 파일시스템은 하드 링크를 지원하지 않는다.
        # 이름을 배타적으로 선점한 뒤 우리가 만든 자리표시자만 교체한다.
        descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
        os.close(descriptor)
        try:
            temporary.replace(destination)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    else:
        temporary.unlink()


def verify_model_runtime_dependencies() -> None:
    """원격 모델 코드의 import를 검증하고 PyInstaller 분석에도 노출한다."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"kornia\.feature\.lightglue",
            )
            import einops
            from kornia.filters import laplacian
    except ImportError as error:
        raise RuntimeError(
            "BiRefNet 실행 구성요소가 누락되었습니다. 개발 환경에서는 "
            "requirements.txt를 다시 설치하세요."
        ) from error

    # 함수 내부 import도 PyInstaller가 수집하며, 참조를 남겨 정적 분석에서
    # 불필요한 import로 오인되지 않게 한다.
    _ = einops, laplacian


def prepare_model_for_device(model, device: str):
    """장치가 지원하는 dtype으로 모델을 옮기고 입력용 dtype을 반환한다."""
    if device == "cpu":
        model = model.float()
    model = model.eval().to(device)
    input_dtype = next(
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    )
    return model, input_dtype


class ToonOutEngine:
    """모델을 한 번 로드하고 여러 이미지에 재사용한다."""

    def __init__(self, model_directory: str | Path | None = None):
        self._model = None
        self._transform = None
        self._torch = None
        self._device = "cpu"
        self._input_dtype = None
        self._model_directory = (
            Path(model_directory)
            if model_directory is not None
            else default_model_directory()
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_label(self) -> str:
        return "NVIDIA GPU" if self._device == "cuda" else "CPU"

    @property
    def model_directory(self) -> Path:
        return self._model_directory

    def set_model_directory(self, directory: str | Path) -> bool:
        """로드 전에는 경로를 바꾸고, 로드 후에는 다음 실행에 맡긴다."""
        if self.is_loaded:
            return False
        self._model_directory = Path(directory)
        return True

    def load(self, report_status: Callable[[str], None] | None = None) -> None:
        if self.is_loaded:
            return

        def report(message: str) -> None:
            if report_status is not None:
                report_status(message)

        report("모델 저장 위치를 확인하는 중")
        self._model_directory.mkdir(parents=True, exist_ok=True)
        configure_huggingface_environment(self._model_directory)
        cache_directory = str(self._model_directory)

        report("필요한 추론 라이브러리를 불러오는 중")

        import torch
        import transformers.configuration_utils
        from huggingface_hub import hf_hub_download
        from torchvision import transforms
        from transformers import AutoConfig, AutoModelForImageSegmentation

        verify_model_runtime_dependencies()

        config_class = transformers.configuration_utils.PretrainedConfig
        if not getattr(config_class, "_toonout_compatibility_patch", False):
            original_getattribute = config_class.__getattribute__

            def patched_getattribute(config, key):
                if key == "is_encoder_decoder":
                    return False
                return original_getattribute(config, key)

            config_class.__getattribute__ = patched_getattribute
            config_class._toonout_compatibility_patch = True

        report("BiRefNet 모델 구조를 준비하는 중")
        config = AutoConfig.from_pretrained(
            BASE_MODEL_REPOSITORY,
            revision=BASE_MODEL_REVISION,
            trust_remote_code=True,
            cache_dir=cache_directory,
        )
        with torch.device("meta"):
            model = AutoModelForImageSegmentation.from_config(
                config,
                trust_remote_code=True,
                code_revision=BASE_MODEL_REVISION,
            )

        report("ToonOut 가중치를 준비하는 중 · 처음에는 시간이 걸릴 수 있습니다")
        checkpoint_path = hf_hub_download(
            repo_id=TOONOUT_REPOSITORY,
            filename=TOONOUT_WEIGHTS,
            revision=TOONOUT_REVISION,
            cache_dir=cache_directory,
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module._orig_mod."):
                key = key[len("module._orig_mod."):]
            elif key.startswith("module."):
                key = key[len("module."):]
            clean_state_dict[key] = value

        model.load_state_dict(clean_state_dict, assign=True)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._input_dtype = prepare_model_for_device(
            model,
            self._device,
        )
        self._torch = torch
        self._transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        report(f"모델 준비 완료 · {self.device_label} 사용")

    def remove_background(self, source_path: str, output_path: str) -> None:
        if not self.is_loaded:
            raise RuntimeError("ToonOut 모델이 준비되지 않았습니다.")

        from PIL import Image, ImageOps
        from torchvision import transforms

        with Image.open(source_path) as opened_image:
            oriented_image = ImageOps.exif_transpose(opened_image)
            source_alpha = _existing_alpha(oriented_image)
            image = oriented_image.convert("RGB")

        input_tensor = self._transform(image).unsqueeze(0).to(
            device=self._device,
            dtype=self._input_dtype,
        )

        with self._torch.inference_mode():
            prediction = self._model(input_tensor)[-1].sigmoid().cpu()

        mask = transforms.ToPILImage()(prediction[0].squeeze())
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        mask = _combine_alpha(mask, source_alpha)

        result = image.copy()
        result.putalpha(mask)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        try:
            result.save(temporary, format="PNG")
            try:
                _publish_without_overwrite(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"같은 이름의 결과 파일이 이미 있습니다: {destination.name}"
                ) from error
        finally:
            temporary.unlink(missing_ok=True)
