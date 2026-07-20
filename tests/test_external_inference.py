import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_worker import emit
from processing import ExternalInferenceThread


class ExternalInferenceTests(unittest.TestCase):
    def test_json_worker_events_match_internal_worker_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_worker.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    import sys

                    parser = argparse.ArgumentParser()
                    parser.add_argument('--model-directory')
                    parser.add_argument('--jobs-file')
                    parser.add_argument('--cancel-file')
                    parser.add_argument('--pause-file')
                    parser.add_argument('--cpu-threads')
                    parser.add_argument('--gpu-memory-fraction')
                    parser.add_argument('--cooldown-ms')
                    args = parser.parse_args()
                    jobs = json.load(open(args.jobs_file, encoding='utf-8'))

                    def emit(event, **values):
                        serialized = (
                            json.dumps(
                                {'event': event, **values},
                                ensure_ascii=False,
                            )
                            + '\\n'
                        )
                        # 기존 PyInstaller worker가 한국어 Windows에서
                        # 실제로 내보내던 바이트를 재현한다.
                        sys.stdout.buffer.write(serialized.encode('cp949'))
                        sys.stdout.buffer.flush()

                    emit('model_status', message='모델 저장 위치를 확인하는 중')
                    emit('model_ready', device_label='NVIDIA GPU')
                    for index, (item_id, source, output) in enumerate(jobs, 1):
                        emit('item_started', item_id=item_id)
                        emit('item_completed', item_id=item_id, output_path=output)
                        emit('progress_changed', completed=index, total=len(jobs))
                    emit('batch_finished', cancelled=False)
                    """
                ),
                encoding="utf-8",
            )
            thread = ExternalInferenceThread(
                [sys.executable, str(script)],
                "C:/models",
            )
            thread.set_jobs([("item-1", "source.png", "result.png")])
            devices = []
            statuses = []
            completed = []
            finished = []
            thread.model_status.connect(statuses.append)
            thread.model_ready.connect(devices.append)
            thread.item_completed.connect(
                lambda item_id, output: completed.append((item_id, output))
            )
            thread.batch_finished.connect(finished.append)

            thread.run()

            self.assertEqual(statuses, ["모델 저장 위치를 확인하는 중"])
            self.assertEqual(devices, ["NVIDIA GPU"])
            self.assertEqual(completed, [("item-1", "result.png")])
            self.assertEqual(finished, [False])

    def test_packaged_worker_protocol_escapes_non_ascii_text(self):
        with patch("builtins.print") as output:
            emit("model_status", message="모델 저장 위치를 확인하는 중")

        serialized = output.call_args.args[0]
        self.assertTrue(serialized.isascii())
        self.assertNotIn("모델", serialized)
        self.assertEqual(
            json.loads(serialized),
            {
                "event": "model_status",
                "message": "모델 저장 위치를 확인하는 중",
            },
        )


if __name__ == "__main__":
    unittest.main()
