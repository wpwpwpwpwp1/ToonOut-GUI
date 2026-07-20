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


class ReleaseManifestTests(unittest.TestCase):
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
