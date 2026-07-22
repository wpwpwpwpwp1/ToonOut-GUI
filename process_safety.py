"""Small cross-process safety helpers used by ToonOut workers."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time


def process_is_running(process_id: int) -> bool:
    """Return whether a process still exists without sending it a signal."""

    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True
    if sys.platform == "win32":
        from ctypes import wintypes

        synchronize = 0x00100000
        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        handle = open_process(
            synchronize | process_query_limited_information,
            False,
            process_id,
        )
        if handle:
            close_handle(handle)
            return True
        # Access denied means that Windows can still see the process. Treat it
        # as active so cleanup code never deletes another process's files.
        return ctypes.get_last_error() == 5

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start_parent_exit_watchdog(parent_process_id: int) -> None:
    """Exit this worker promptly if its owning ToonOut process disappears."""

    if parent_process_id <= 0 or parent_process_id == os.getpid():
        return

    def watch_parent() -> None:
        if sys.platform == "win32":
            from ctypes import wintypes

            synchronize = 0x00100000
            infinite = 0xFFFFFFFF
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            open_process.restype = wintypes.HANDLE
            wait_for_single_object = kernel32.WaitForSingleObject
            wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            wait_for_single_object.restype = wintypes.DWORD
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            handle = open_process(synchronize, False, parent_process_id)
            if handle:
                try:
                    wait_for_single_object(handle, infinite)
                finally:
                    close_handle(handle)
                os._exit(3)

        # Portable fallback, and a Windows fallback for unusual security
        # policies that deny a synchronization handle.
        while process_is_running(parent_process_id):
            time.sleep(0.5)
        os._exit(3)

    threading.Thread(
        target=watch_parent,
        name="toonout-parent-watchdog",
        daemon=True,
    ).start()


class KillOnCloseProcessJob:
    """Windows job object that kills a child when the parent handle closes."""

    def __init__(self) -> None:
        self._handle = None

    def assign(self, process) -> None:
        if sys.platform != "win32":
            return

        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job_object_extended_limit_information = 9
        job_object_limit_kill_on_job_close = 0x00002000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_job_object = kernel32.CreateJobObjectW
        create_job_object.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        create_job_object.restype = wintypes.HANDLE
        set_information = kernel32.SetInformationJobObject
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        assign_process = kernel32.AssignProcessToJobObject
        assign_process.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        assign_process.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        job_handle = create_job_object(None, None)
        if not job_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            information = ExtendedLimitInformation()
            information.BasicLimitInformation.LimitFlags = (
                job_object_limit_kill_on_job_close
            )
            if not set_information(
                job_handle,
                job_object_extended_limit_information,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = wintypes.HANDLE(int(process._handle))
            if not assign_process(job_handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            close_handle(job_handle)
            raise
        self._handle = job_handle

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None or sys.platform != "win32":
            return
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)

    def __enter__(self) -> "KillOnCloseProcessJob":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
