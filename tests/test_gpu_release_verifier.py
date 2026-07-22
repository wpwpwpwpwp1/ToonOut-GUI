import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_gpu_pack_release import verify_release_assets


class GpuReleaseVerifierTests(unittest.TestCase):
    def _fixture(self, root: Path):
        part_content = b"gpu-pack-part"
        part_hash = hashlib.sha256(part_content).hexdigest()
        manifest_document = json.dumps(
            {
                "schema_version": 1,
                "kind": "toonout-nvidia-gpu-pack-parts",
                "archive_filename": "ToonOut-NVIDIA-GPU-Pack.zip",
                "archive_size": len(part_content),
                "archive_sha256": part_hash,
                "parts": [
                    {
                        "filename": "ToonOut-NVIDIA-GPU-Pack.part01",
                        "size": len(part_content),
                        "sha256": part_hash,
                    }
                ],
            }
        ).encode("utf-8")
        manifest = root / "ToonOut-NVIDIA-GPU-Pack.parts.json"
        manifest.write_bytes(manifest_document)
        manifest_hash = hashlib.sha256(manifest_document).hexdigest()
        assets = {
            "assets": [
                {
                    "name": manifest.name,
                    "size": len(manifest_document),
                    "digest": f"sha256:{manifest_hash}",
                    "state": "uploaded",
                },
                {
                    "name": "ToonOut-NVIDIA-GPU-Pack.part01",
                    "size": len(part_content),
                    "digest": f"sha256:{part_hash}",
                    "state": "uploaded",
                },
            ]
        }
        return manifest, manifest_document, assets

    def test_complete_release_assets_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, document, assets = self._fixture(Path(directory))
            with (
                patch(
                    "scripts.verify_gpu_pack_release.GPU_PACK_MANIFEST_SIZE",
                    len(document),
                ),
                patch(
                    "scripts.verify_gpu_pack_release.GPU_PACK_MANIFEST_SHA256",
                    hashlib.sha256(document).hexdigest(),
                ),
                patch(
                    "scripts.verify_gpu_pack_release.GPU_PACK_ARCHIVE_SIZE",
                    len(b"gpu-pack-part"),
                ),
            ):
                verified = verify_release_assets(manifest, assets)

            self.assertEqual(
                set(verified),
                {
                    "ToonOut-NVIDIA-GPU-Pack.parts.json",
                    "ToonOut-NVIDIA-GPU-Pack.part01",
                },
            )

    def test_missing_part_blocks_release(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, document, assets = self._fixture(Path(directory))
            assets["assets"].pop()
            with (
                patch(
                    "scripts.verify_gpu_pack_release.GPU_PACK_MANIFEST_SIZE",
                    len(document),
                ),
                patch(
                    "scripts.verify_gpu_pack_release.GPU_PACK_MANIFEST_SHA256",
                    hashlib.sha256(document).hexdigest(),
                ),
                patch(
                    "scripts.verify_gpu_pack_release.GPU_PACK_ARCHIVE_SIZE",
                    len(b"gpu-pack-part"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "자산이 없습니다"):
                    verify_release_assets(manifest, assets)


if __name__ == "__main__":
    unittest.main()
