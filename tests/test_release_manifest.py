import base64
import json
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from scripts.create_update_manifest import create_manifest
from update_service import verify_signed_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseManifestTests(unittest.TestCase):
    def test_installer_replaces_old_payload_and_cleans_download_cache(self):
        script = (PROJECT_ROOT / "installer" / "ToonOut.iss").read_text(
            encoding="utf-8"
        )
        build_script = (PROJECT_ROOT / "scripts" / "build_app.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("[InstallDelete]", script)
        self.assertIn('Name: "{app}\\_internal"', script)
        self.assertIn("--cleanup-update-cache", script)
        self.assertIn('Name: "{localappdata}\\ToonOut\\updates"', script)
        self.assertIn('$BrandIconPath', build_script)
        self.assertIn('$env:PYTHONNOUSERSITE = "1"', build_script)
        self.assertIn('$env:PYTHONUSERBASE = Join-Path $BuildRoot', build_script)

        gpu_build_script = (
            PROJECT_ROOT / "scripts" / "build_gpu_pack.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('$env:PYTHONNOUSERSITE = "1"', gpu_build_script)
        self.assertIn(
            '$env:PYTHONUSERBASE = Join-Path $BuildRoot', gpu_build_script
        )

        workflow = (
            PROJECT_ROOT / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("Verify pinned GPU pack release", workflow)
        self.assertIn("verify_gpu_pack_release.py", workflow)

    def test_release_script_creates_runtime_verifiable_manifest(self):
        private_key = Ed25519PrivateKey.generate()
        private_value = base64.b64encode(
            private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        ).decode("ascii")
        public_value = base64.b64encode(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii")

        with tempfile.TemporaryDirectory() as directory:
            installer = Path(directory) / "ToonOut-Setup-0.2.0.exe"
            installer.write_bytes(b"signed setup")

            envelope = create_manifest(
                installer,
                "0.2.0",
                "0.1.0",
                "owner/ToonOut-Interface",
                "v0.2.0",
                private_value,
            )
            release = verify_signed_manifest(
                json.dumps(envelope).encode("utf-8"),
                public_value,
            )

            self.assertEqual(release.version, "0.2.0")
            self.assertEqual(release.installer_size, len(b"signed setup"))
            self.assertIn("/v0.2.0/", release.installer_url)


if __name__ == "__main__":
    unittest.main()
