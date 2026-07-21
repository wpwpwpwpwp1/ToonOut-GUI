"""업데이트 네트워크·파일 작업을 UI 스레드 밖에서 실행한다."""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from update_service import (
    UpdateCancelled,
    UpdateError,
    UpdateRelease,
    download_installer,
    fetch_update_release,
    is_newer_version,
    prune_update_cache,
)


class UpdateCheckThread(QThread):
    available = Signal(object)
    up_to_date = Signal()
    failed = Signal(str)

    def __init__(
        self,
        manifest_url: str,
        public_key_b64: str,
        current_version: str,
        update_root: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._manifest_url = manifest_url
        self._public_key_b64 = public_key_b64
        self._current_version = current_version
        self._update_root = Path(update_root) if update_root is not None else None

    def run(self):
        try:
            if self._update_root is not None:
                prune_update_cache(self._update_root, None)
            release = fetch_update_release(
                self._manifest_url,
                self._public_key_b64,
            )
            if is_newer_version(release.version, self._current_version):
                self.available.emit(release)
            else:
                self.up_to_date.emit()
        except UpdateError as error:
            self.failed.emit(str(error))


class UpdateDownloadThread(QThread):
    progress_changed = Signal(int, int)
    ready = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        release: UpdateRelease,
        destination_directory: str | Path,
        parent=None,
    ):
        super().__init__(parent)
        self._release = release
        self._destination_directory = Path(destination_directory)

    def run(self):
        try:
            prune_update_cache(
                self._destination_directory.parent,
                self._release.version,
            )
            installer = download_installer(
                self._release,
                self._destination_directory,
                progress=self.progress_changed.emit,
                cancelled=self.isInterruptionRequested,
            )
            self.ready.emit(str(installer))
        except UpdateCancelled:
            return
        except UpdateError as error:
            self.failed.emit(str(error))
