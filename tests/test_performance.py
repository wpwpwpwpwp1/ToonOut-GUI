import threading
import time
import unittest

from performance import (
    PerformanceMode,
    PerformancePolicy,
    apply_torch_policy,
    preset_policy,
)
from processing import InferenceThread


class PerformancePolicyTests(unittest.TestCase):
    def test_presets_trade_speed_for_lower_resource_use(self):
        eco = preset_policy(PerformanceMode.ECO, logical_cpus=8)
        balanced = preset_policy(PerformanceMode.BALANCED, logical_cpus=8)
        maximum = preset_policy(PerformanceMode.MAXIMUM, logical_cpus=8)

        self.assertEqual(eco.cpu_threads, 2)
        self.assertEqual(balanced.cpu_threads, 4)
        self.assertEqual(maximum.cpu_threads, 8)
        self.assertGreater(eco.cooldown_ms, balanced.cooldown_ms)
        self.assertGreater(balanced.cooldown_ms, maximum.cooldown_ms)
        self.assertLess(eco.gpu_memory_fraction, maximum.gpu_memory_fraction)

    def test_torch_policy_applies_cpu_and_gpu_memory_limits(self):
        class FakeCuda:
            memory_fraction = None

            @staticmethod
            def is_available():
                return True

            @classmethod
            def set_per_process_memory_fraction(cls, value):
                cls.memory_fraction = value

        class FakeTorch:
            cuda = FakeCuda()
            threads = None
            interop_threads = None

            @classmethod
            def set_num_threads(cls, value):
                cls.threads = value

            @classmethod
            def set_num_interop_threads(cls, value):
                cls.interop_threads = value

        apply_torch_policy(
            FakeTorch,
            PerformancePolicy(PerformanceMode.CUSTOM, 2, 0.75, 300),
        )

        self.assertEqual(FakeTorch.threads, 2)
        self.assertEqual(FakeTorch.interop_threads, 1)
        self.assertEqual(FakeCuda.memory_fraction, 0.75)


class PauseResumeTests(unittest.TestCase):
    def test_pause_waits_at_an_image_boundary_and_can_resume(self):
        first_started = threading.Event()
        finish_first = threading.Event()
        second_started = threading.Event()

        class FakeEngine:
            is_loaded = True
            device_label = "CPU"

            @staticmethod
            def load(_report):
                return None

            @staticmethod
            def remove_background(source, _output):
                if source == "first.png":
                    first_started.set()
                    finish_first.wait(2)
                else:
                    second_started.set()

        worker = InferenceThread(
            performance_policy=PerformancePolicy(
                PerformanceMode.CUSTOM,
                1,
                1.0,
                0,
            )
        )
        worker._engine = FakeEngine()
        worker.set_jobs(
            [
                ("first", "first.png", "first-out.png"),
                ("second", "second.png", "second-out.png"),
            ]
        )
        worker.start()
        self.assertTrue(first_started.wait(2))
        worker.request_pause()
        finish_first.set()
        time.sleep(0.2)
        self.assertTrue(worker.isRunning())
        self.assertFalse(second_started.is_set())

        worker.request_resume()
        self.assertTrue(worker.wait(2_000))
        self.assertTrue(second_started.is_set())


if __name__ == "__main__":
    unittest.main()
