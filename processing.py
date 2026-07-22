"""UI를 멈추지 않고 ToonOut 배치와 로컬 파일 작업을 처리한다."""

import json
import locale
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QThread, QTimer, Signal

from acceleration import detect_acceleration
from gpu_download import GpuPackRelease, download_and_install_gpu_runtime
from gpu_runtime import (
    GpuRuntimeCancelled,
    cleanup_gpu_runtime_artifacts,
    delete_gpu_runtime,
    gpu_runtime_size,
    install_gpu_runtime,
    load_gpu_runtime_manifest,
)
from inference import ToonOutEngine
from model_installation import (
    delete_model_files,
    model_is_installed,
    move_model_files,
    prepare_model_cache_for_install,
)
from performance import (
    PerformanceMode,
    PerformancePolicy,
    apply_torch_policy,
    preset_policy,
)
from process_safety import KillOnCloseProcessJob


class InferenceThread(QThread):
    model_status = Signal(str)
    model_ready = Signal(str)
    model_failed = Signal(str)
    item_started = Signal(str)
    item_completed = Signal(str, str)
    item_failed = Signal(str, str)
    progress_changed = Signal(int, int)
    pause_reached = Signal()
    batch_finished = Signal(bool)

    def __init__(
        self,
        model_directory: str | None = None,
        performance_policy: PerformancePolicy | None = None,
    ):
        super().__init__()
        self._engine = ToonOutEngine(model_directory)
        self._jobs: list[tuple[str, str, str]] = []
        self._cancel_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._performance_policy = (
            performance_policy
            or preset_policy(PerformanceMode.BALANCED)
        ).normalized()

    @property
    def job_count(self) -> int:
        return len(self._jobs)

    @property
    def model_is_loaded(self) -> bool:
        return self._engine.is_loaded

    def set_model_directory(self, directory: str) -> bool:
        if self.isRunning():
            raise RuntimeError("모델 처리 중에는 저장 위치를 바꿀 수 없습니다.")
        return self._engine.set_model_directory(directory)

    def reset_engine(self, directory: str) -> None:
        if self.isRunning():
            raise RuntimeError("모델 처리 중에는 모델을 초기화할 수 없습니다.")
        self._engine = ToonOutEngine(directory)

    def set_jobs(self, jobs: list[tuple[str, str, str]]):
        if self.isRunning():
            raise RuntimeError("이미 추론 작업이 실행 중입니다.")
        self._jobs = list(jobs)
        self._cancel_event.clear()
        self._resume_event.set()

    def set_performance_policy(self, policy: PerformancePolicy) -> None:
        if self.isRunning():
            raise RuntimeError("처리 중에는 성능 모드를 바꿀 수 없습니다.")
        self._performance_policy = policy.normalized()

    def request_cancel(self):
        self._cancel_event.set()
        self._resume_event.set()

    def request_pause(self):
        self._resume_event.clear()

    def request_resume(self):
        self._resume_event.set()

    def _wait_before_next_item(self, last_completed_at: float | None) -> bool:
        pause_announced = False
        while True:
            if self._cancel_event.is_set():
                return False
            if not self._resume_event.is_set():
                if not pause_announced:
                    self.pause_reached.emit()
                    pause_announced = True
                self._resume_event.wait(0.1)
                continue

            if last_completed_at is None:
                return True
            elapsed_ms = (time.monotonic() - last_completed_at) * 1000
            remaining_ms = self._performance_policy.cooldown_ms - elapsed_ms
            if remaining_ms <= 0:
                return True
            self._cancel_event.wait(min(0.1, remaining_ms / 1000))

    @staticmethod
    def friendly_error(error: Exception) -> str:
        message = str(error).strip()
        lowered = message.lower()
        winerror = getattr(error, "winerror", None)
        errno = getattr(error, "errno", None)

        if "out of memory" in lowered:
            return (
                "메모리가 부족합니다. 성능 모드의 메모리 상한을 높이거나 "
                "더 작은 이미지로 다시 시도하세요."
            )
        if (
            winerror == 448
            or "untrusted mount point" in lowered
            or "신뢰할 수 없는 탑재 지점" in message
            or "Windows 링크가 남아" in message
        ):
            return (
                "Windows가 이전 모델 캐시의 링크를 차단했습니다. "
                "ToonOut이 남은 캐시를 정리한 뒤에는 링크 없는 일반 파일로 "
                "다시 설치합니다. 계속되면 기본 모델 저장 폴더를 선택하세요."
            )
        if winerror == 206 or "filename or extension is too long" in lowered:
            return (
                "모델 저장 경로가 너무 깁니다. 드라이브 루트에 가까운 짧은 "
                "폴더(예: D:\\ToonOutModels)를 선택하세요."
            )
        if winerror in {112, 1816} or errno == 28 or "no space" in lowered:
            return "저장 공간이 부족합니다. 여유 공간을 확보하세요."
        if winerror in {32, 33} or "used by another process" in lowered:
            return (
                "다른 프로그램이 모델 파일을 사용 중입니다. 실행 중인 ToonOut "
                "창과 백신 검사를 확인한 뒤 다시 시도하세요."
            )
        if "ssl" in lowered or "certificate" in lowered:
            return (
                "보안 연결을 확인하지 못했습니다. Windows 날짜·시간과 "
                "회사/학교 네트워크의 HTTPS 인증서를 확인하세요."
            )
        if (
            "connection" in lowered
            or "network" in lowered
            or "proxy" in lowered
            or "timed out" in lowered
            or "name resolution" in lowered
            or "outgoing traffic" in lowered
        ):
            return (
                "모델 파일을 받지 못했습니다. 인터넷 연결과 프록시·방화벽을 "
                "확인한 뒤 다시 시도하세요."
            )
        if (
            "not found in your environment" in lowered
            or "실행 구성요소가 누락" in message
        ):
            return (
                "모델 실행 구성요소가 누락되었습니다. 앱을 다시 설치하거나, "
                "개발 환경에서는 requirements.txt를 다시 설치하세요."
            )
        if (
            isinstance(error, PermissionError)
            or "permission" in lowered
            or "access is denied" in lowered
            or "액세스가 거부" in message
        ):
            return (
                "모델 저장 폴더에 쓸 수 없습니다. "
                "모델 설치 창에서 다른 폴더를 선택하세요."
            )
        if (
            "failed finding central directory" in lowered
            or "invalid header" in lowered
            or "corrupt" in lowered
        ):
            return (
                "내려받은 모델 파일이 손상되었습니다. 남은 파일을 정리한 뒤 "
                "모델 설치를 다시 시도하세요."
            )
        if not message:
            return "알 수 없는 오류가 발생했습니다."
        return message[:240]

    @staticmethod
    def friendly_model_install_error(error: Exception) -> str:
        """Map install failures without exposing raw user paths in the UI."""

        friendly = InferenceThread.friendly_error(error)
        raw_preview = str(error).strip()[:240]
        if friendly != raw_preview:
            return friendly
        return (
            "모델 설치 중 예상하지 못한 오류가 발생했습니다. 남은 파일을 "
            "정리한 뒤 다시 시도하세요. 계속되면 모델 저장 위치를 바꾸거나 "
            f"앱을 다시 설치하세요. 오류 종류: {type(error).__name__}"
        )

    def run(self):
        try:
            import torch

            apply_torch_policy(torch, self._performance_policy)
            self._engine.load(self.model_status.emit)
            self.model_ready.emit(self._engine.device_label)
        except Exception as error:
            self.model_failed.emit(self.friendly_error(error))
            return

        total = len(self._jobs)
        completed_count = 0
        last_completed_at: float | None = None

        for item_id, source_path, output_path in self._jobs:
            if not self._wait_before_next_item(last_completed_at):
                break

            self.item_started.emit(item_id)
            try:
                self._engine.remove_background(source_path, output_path)
                self.item_completed.emit(item_id, output_path)
            except Exception as error:
                self.item_failed.emit(item_id, self.friendly_error(error))

            completed_count += 1
            self.progress_changed.emit(completed_count, total)
            last_completed_at = time.monotonic()

        self.batch_finished.emit(self._cancel_event.is_set())


def run_model_install_worker(model_directory: str, status_path: str) -> int:
    """별도 프로세스에서 모델을 설치하고 JSON-lines 상태만 부모에 남긴다."""

    destination = Path(status_path)

    def emit(event: str, **payload) -> None:
        record = json.dumps(
            {"event": event, **payload},
            ensure_ascii=False,
        )
        with destination.open("a", encoding="utf-8") as status_file:
            status_file.write(f"{record}\n")
            status_file.flush()

    try:
        prepare_model_cache_for_install(
            model_directory,
            lambda message: emit("status", message=message),
        )
        engine = ToonOutEngine(model_directory)
        engine.load(
            lambda message: emit("status", message=message),
            lambda status, percent: emit(
                "progress",
                status=status,
                percent=percent,
            ),
        )
        emit("success")
        return 0
    except Exception as error:
        emit(
            "error",
            message=InferenceThread.friendly_model_install_error(error),
        )
        return 1


def run_model_cleanup_worker(model_directory: str) -> int:
    """실패하거나 취소한 설치가 남긴 앱 관리 모델 파일만 삭제한다."""

    try:
        delete_model_files(model_directory)
        return 0
    except (OSError, ValueError):
        return 1


class ModelInstallProcess(QProcess):
    """취소할 수 있는 별도 모델 설치 프로세스와 상태 파일을 관리한다."""

    status_changed = Signal(str)
    progress_changed = Signal(str, int)
    installation_succeeded = Signal()
    installation_failed = Signal(str)
    installation_cancelled = Signal()

    def __init__(self, model_directory: str, parent=None):
        super().__init__(parent)
        self._model_directory = model_directory
        descriptor, status_path = tempfile.mkstemp(
            prefix="toonout-model-install-",
            suffix=".jsonl",
        )
        os.close(descriptor)
        self._status_path = Path(status_path)
        self._status_offset = 0
        self._status_buffer = ""
        self._last_error: str | None = None
        self._cancel_requested = False
        self._cleaning_up = False
        self._cleanup_outcome = "failure"
        self._settled = False

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._read_status)
        self.finished.connect(self._on_finished)
        self.errorOccurred.connect(self._on_process_error)

    def start_installation(self) -> None:
        worker_arguments = [
            "--model-install-worker",
            self._model_directory,
            "--model-install-status",
            str(self._status_path),
        ]
        self._set_worker_command(worker_arguments)
        self._status_timer.start()
        self.start()

    def _set_worker_command(self, worker_arguments: list[str]) -> None:
        worker_arguments = [
            *worker_arguments,
            "--parent-pid",
            str(os.getpid()),
        ]
        if getattr(sys, "frozen", False):
            program = sys.executable
            arguments = worker_arguments
        else:
            program = sys.executable
            arguments = [
                str(Path(__file__).with_name("main.py").resolve()),
                *worker_arguments,
            ]
        self.setProgram(program)
        self.setArguments(arguments)

    def request_cancel(self) -> None:
        if self._cleaning_up or self.state() == QProcess.ProcessState.NotRunning:
            return
        self._cancel_requested = True
        self.status_changed.emit("모델 설치를 취소하는 중")
        self.terminate()
        QTimer.singleShot(2_000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if (
            not self._cleaning_up
            and self.state() != QProcess.ProcessState.NotRunning
        ):
            self.kill()

    def _read_status(self, final: bool = False) -> None:
        try:
            with self._status_path.open("rb") as status_file:
                status_file.seek(self._status_offset)
                chunk = status_file.read()
                self._status_offset = status_file.tell()
        except OSError:
            return
        if chunk:
            self._status_buffer += chunk.decode("utf-8", errors="replace")
        lines = self._status_buffer.split("\n")
        self._status_buffer = lines.pop()
        if final and self._status_buffer:
            lines.append(self._status_buffer)
            self._status_buffer = ""
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "status":
                self.status_changed.emit(str(record.get("message", "")))
            elif event == "progress":
                self.progress_changed.emit(
                    str(record.get("status", "모델 설치 중")),
                    max(0, min(100, int(record.get("percent", 0)))),
                )
            elif event == "error":
                self._last_error = str(
                    record.get("message", "모델을 설치하지 못했습니다.")
                )

    def _on_process_error(self, error) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        if self._cleaning_up:
            self._finish_cleanup(False)
        elif not self._cancel_requested:
            self._settle_failure("모델 설치 작업을 시작하지 못했습니다.")

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        if self._settled:
            return
        if self._cleaning_up:
            self._finish_cleanup(exit_code == 0)
            return
        self._status_timer.stop()
        self._read_status(final=True)
        if exit_code == 0 and model_is_installed(self._model_directory):
            self._settled = True
            self.installation_succeeded.emit()
            self._remove_status_file()
            return

        self._cleanup_outcome = (
            "cancelled" if self._cancel_requested else "failure"
        )
        if self._last_error is None and self._cleanup_outcome == "failure":
            self._last_error = (
                "모델 설치를 완료하지 못했습니다. 다시 시도하세요."
            )
        self._start_cleanup_worker()

    def _start_cleanup_worker(self) -> None:
        self._cleaning_up = True
        self.status_changed.emit("남은 모델 파일을 정리하는 중")
        self._set_worker_command(
            ["--model-cleanup-worker", self._model_directory]
        )
        self.start()

    def _finish_cleanup(self, succeeded: bool) -> None:
        self._cleaning_up = False
        if not succeeded:
            message = self._last_error or "모델 설치를 취소했습니다."
            self._settle_failure(
                f"{message} 남은 모델 파일을 자동으로 정리하지 못했습니다. "
                "모델 저장 폴더에서 다시 시도하세요."
            )
            return

        if self._cleanup_outcome == "cancelled":
            self._settled = True
            self.installation_cancelled.emit()
            self._remove_status_file()
            return
        self._settle_failure(
            self._last_error or "모델 설치를 완료하지 못했습니다. 다시 시도하세요."
        )

    def _settle_failure(self, message: str) -> None:
        if self._settled:
            return
        self._settled = True
        self._status_timer.stop()
        self.installation_failed.emit(message)
        self._remove_status_file()

    def _remove_status_file(self) -> None:
        try:
            self._status_path.unlink(missing_ok=True)
        except OSError:
            pass


class ExternalInferenceThread(QThread):
    """GPU 팩의 worker를 실행하되 기존 추론 스레드와 같은 신호를 낸다."""

    model_status = Signal(str)
    model_ready = Signal(str)
    model_failed = Signal(str)
    item_started = Signal(str)
    item_completed = Signal(str, str)
    item_failed = Signal(str, str)
    progress_changed = Signal(int, int)
    pause_reached = Signal()
    batch_finished = Signal(bool)

    def __init__(
        self,
        worker_command: list[str],
        model_directory: str | None = None,
        performance_policy: PerformancePolicy | None = None,
    ):
        super().__init__()
        self._worker_command = list(worker_command)
        self._model_directory = model_directory or ""
        self._jobs: list[tuple[str, str, str]] = []
        self._cancel_event = threading.Event()
        self._cancel_path: Path | None = None
        self._pause_requested = threading.Event()
        self._pause_path: Path | None = None
        self._performance_policy = (
            performance_policy
            or preset_policy(PerformanceMode.BALANCED)
        ).normalized()

    @property
    def job_count(self) -> int:
        return len(self._jobs)

    @property
    def model_is_loaded(self) -> bool:
        # 별도 프로세스는 배치 종료 시 메모리에서 내려간다.
        return False

    def set_model_directory(self, directory: str) -> bool:
        if self.isRunning():
            raise RuntimeError("모델 처리 중에는 저장 위치를 바꿀 수 없습니다.")
        self._model_directory = directory
        return True

    def reset_engine(self, directory: str) -> None:
        self.set_model_directory(directory)

    def set_jobs(self, jobs: list[tuple[str, str, str]]):
        if self.isRunning():
            raise RuntimeError("이미 추론 작업이 실행 중입니다.")
        self._jobs = list(jobs)
        self._cancel_event.clear()
        self._pause_requested.clear()

    def set_performance_policy(self, policy: PerformancePolicy) -> None:
        if self.isRunning():
            raise RuntimeError("처리 중에는 성능 모드를 바꿀 수 없습니다.")
        self._performance_policy = policy.normalized()

    def request_cancel(self):
        self._cancel_event.set()
        cancel_path = self._cancel_path
        if cancel_path is not None:
            try:
                cancel_path.touch(exist_ok=True)
            except OSError:
                pass

    def request_pause(self):
        self._pause_requested.set()
        pause_path = self._pause_path
        if pause_path is not None:
            try:
                pause_path.touch(exist_ok=True)
            except OSError:
                pass

    def request_resume(self):
        self._pause_requested.clear()
        pause_path = self._pause_path
        if pause_path is not None:
            try:
                pause_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _dispatch_record(self, record: dict) -> bool:
        event = record.get("event")
        if event == "model_status":
            self.model_status.emit(str(record.get("message", "")))
        elif event == "model_ready":
            self.model_ready.emit(str(record.get("device_label", "NVIDIA GPU")))
        elif event == "model_failed":
            self.model_failed.emit(str(record.get("error", "GPU 모델 준비 실패")))
            return True
        elif event == "item_started":
            self.item_started.emit(str(record.get("item_id", "")))
        elif event == "item_completed":
            self.item_completed.emit(
                str(record.get("item_id", "")),
                str(record.get("output_path", "")),
            )
        elif event == "item_failed":
            self.item_failed.emit(
                str(record.get("item_id", "")),
                str(record.get("error", "GPU 처리 실패")),
            )
        elif event == "progress_changed":
            self.progress_changed.emit(
                int(record.get("completed", 0)),
                int(record.get("total", 0)),
            )
        elif event == "batch_paused":
            self.pause_reached.emit()
        elif event == "batch_finished":
            self.batch_finished.emit(bool(record.get("cancelled", False)))
            return True
        return False

    @staticmethod
    def _decode_worker_line(raw_line: bytes) -> str:
        """새 UTF-8 worker와 기존 Windows 코드페이지 worker를 모두 읽는다."""
        encodings = ["utf-8", locale.getpreferredencoding(False)]
        if sys.platform == "win32":
            # 한국어 Windows에서 만들어진 기존 GPU 팩과의 호환 경로다.
            encodings.append("cp949")

        tried: set[str] = set()
        for encoding in encodings:
            normalized = encoding.lower()
            if normalized in tried:
                continue
            tried.add(normalized)
            try:
                return raw_line.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return raw_line.decode("utf-8", errors="replace")

    def run(self):
        terminal_event_received = False
        diagnostic_lines: list[str] = []
        with tempfile.TemporaryDirectory(prefix="toonout-gpu-job-") as directory:
            working_directory = Path(directory)
            jobs_path = working_directory / "jobs.json"
            cancel_path = working_directory / "cancel"
            pause_path = working_directory / "pause"
            self._cancel_path = cancel_path
            self._pause_path = pause_path
            jobs_path.write_text(
                json.dumps(self._jobs, ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                *self._worker_command,
                "--model-directory",
                self._model_directory,
                "--jobs-file",
                str(jobs_path),
                "--cancel-file",
                str(cancel_path),
                "--pause-file",
                str(pause_path),
                "--cpu-threads",
                str(self._performance_policy.cpu_threads),
                "--gpu-memory-fraction",
                str(self._performance_policy.gpu_memory_fraction),
                "--cooldown-ms",
                str(self._performance_policy.cooldown_ms),
            ]
            creation_flags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0
            )
            worker_environment = os.environ.copy()
            # Python worker에는 UTF-8을 우선 요청한다. 이 환경 변수를 무시하는
            # 기존 frozen worker는 아래의 바이트 디코더가 별도로 처리한다.
            worker_environment["PYTHONIOENCODING"] = "utf-8"
            worker_environment["PYTHONUTF8"] = "1"
            process = None
            process_job = KillOnCloseProcessJob()
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                    env=worker_environment,
                )
                try:
                    process_job.assign(process)
                except OSError as error:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise OSError(
                        "GPU worker의 안전 종료를 구성하지 못했습니다."
                    ) from error
                if self._cancel_event.is_set():
                    cancel_path.touch(exist_ok=True)
                if self._pause_requested.is_set():
                    pause_path.touch(exist_ok=True)
                assert process.stdout is not None
                try:
                    for raw_line in process.stdout:
                        line = self._decode_worker_line(raw_line).strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            diagnostic_lines.append(line)
                            diagnostic_lines = diagnostic_lines[-8:]
                            continue
                        if (
                            isinstance(record, dict)
                            and self._dispatch_record(record)
                        ):
                            terminal_event_received = True
                finally:
                    process.stdout.close()
                return_code = process.wait()
            except OSError as error:
                self.model_failed.emit(
                    f"GPU worker를 실행하지 못했습니다: {error}"
                )
                return
            finally:
                process_job.close()
                self._cancel_path = None
                self._pause_path = None

        if terminal_event_received:
            return
        if self._cancel_event.is_set():
            self.batch_finished.emit(True)
            return
        details = " · ".join(diagnostic_lines[-3:])
        message = f"GPU worker가 예기치 않게 종료되었습니다 (코드 {return_code})."
        if details:
            message = f"{message}\n{details[:500]}"
        self.model_failed.emit(message)


class ModelFileThread(QThread):
    status_changed = Signal(str)
    succeeded = Signal(bool)
    failed = Signal(str)

    def __init__(
        self,
        action: str,
        source_directory: str,
        destination_directory: str | None = None,
    ):
        super().__init__()
        self._action = action
        self._source_directory = source_directory
        self._destination_directory = destination_directory

    def run(self):
        try:
            if self._action == "delete":
                delete_model_files(
                    self._source_directory,
                    self.status_changed.emit,
                )
                self.succeeded.emit(True)
                return
            if self._action == "move" and self._destination_directory:
                source_removed = move_model_files(
                    self._source_directory,
                    self._destination_directory,
                    self.status_changed.emit,
                )
                self.succeeded.emit(source_removed)
                return
            raise ValueError("지원하지 않는 모델 파일 작업입니다.")
        except Exception as error:
            self.failed.emit(InferenceThread.friendly_error(error))


class GpuRuntimeFileThread(QThread):
    status_changed = Signal(str)
    progress_changed = Signal(str, int)
    installed = Signal(object)
    deleted = Signal()
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        action: str,
        runtime_directory: str,
        pack_path: str | None = None,
        pack_release: GpuPackRelease | None = None,
    ):
        super().__init__()
        self._action = action
        self._runtime_directory = runtime_directory
        self._pack_path = pack_path
        self._pack_release = pack_release
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self):
        try:
            if self._action == "install":
                if self._pack_release is not None:
                    manifest = download_and_install_gpu_runtime(
                        self._pack_release,
                        self._runtime_directory,
                        report=self.status_changed.emit,
                        cancelled=self._cancel_event.is_set,
                        report_progress=self.progress_changed.emit,
                    )
                elif self._pack_path:
                    manifest = install_gpu_runtime(
                        self._pack_path,
                        self._runtime_directory,
                        report=self.status_changed.emit,
                        should_cancel=self._cancel_event.is_set,
                        report_progress=self.progress_changed.emit,
                    )
                else:
                    raise ValueError("GPU 가속 팩 배포 정보가 없습니다.")
                self.installed.emit(manifest)
                return
            if self._action == "delete":
                self.status_changed.emit("GPU 가속 파일을 삭제하는 중")
                delete_gpu_runtime(self._runtime_directory)
                self.deleted.emit()
                return
            raise ValueError("지원하지 않는 GPU 가속 팩 작업입니다.")
        except GpuRuntimeCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(InferenceThread.friendly_error(error))


class AccelerationDetectionThread(QThread):
    detected = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        runtime_directory: str,
        gpu_enabled: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._runtime_directory = runtime_directory
        self._gpu_enabled = gpu_enabled

    def run(self):
        try:
            cleanup_gpu_runtime_artifacts(self._runtime_directory)
            try:
                runtime = load_gpu_runtime_manifest(
                    self._runtime_directory,
                    verify_files=False,
                )
            except Exception:
                runtime = None
            info = detect_acceleration(
                runtime,
                self._gpu_enabled,
                runtime_size=gpu_runtime_size(self._runtime_directory),
                runtime_directory=self._runtime_directory,
            )
            self.detected.emit(info)
        except Exception as error:
            self.failed.emit(str(error) or "처리 장치를 확인하지 못했습니다.")
