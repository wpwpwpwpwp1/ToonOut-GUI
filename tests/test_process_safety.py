import os
import subprocess
import sys
import unittest

from process_safety import KillOnCloseProcessJob, process_is_running


class ProcessSafetyTests(unittest.TestCase):
    def test_current_process_is_running_and_invalid_ids_are_not(self):
        self.assertTrue(process_is_running(os.getpid()))
        self.assertFalse(process_is_running(0))
        self.assertFalse(process_is_running(-1))

    @unittest.skipUnless(sys.platform == "win32", "Windows job object test")
    def test_closing_job_object_terminates_assigned_worker(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        job = KillOnCloseProcessJob()
        try:
            job.assign(process)
            job.close()
            process.wait(timeout=5)
            self.assertIsNotNone(process.returncode)
        finally:
            job.close()
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
