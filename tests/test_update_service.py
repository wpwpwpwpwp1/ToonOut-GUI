import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from update_service import (
    UpdateConfigurationError,
    UpdateRelease,
    UpdateSecurityError,
    canonical_payload_bytes,
    download_installer,
    is_newer_version,
    prune_update_cache,
    verify_signed_manifest,
)


class FakeResponse:
    def __init__(self, content: bytes):
        self._content = content
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._content) - self._offset
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def signed_document(payload: dict[str, object]):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signature = private_key.sign(canonical_payload_bytes(payload))
    document = json.dumps(
        {
            "payload": payload,
            "signature": base64.b64encode(signature).decode("ascii"),
        }
    ).encode("utf-8")
    return document, base64.b64encode(public_key).decode("ascii")


def valid_payload(content: bytes = b"installer") -> dict[str, object]:
    return {
        "schema_version": 1,
        "version": "0.2.0",
        "minimum_supported_version": "0.1.0",
        "installer_url": "https://github.com/example/release/setup.exe",
        "installer_name": "ToonOut-Setup-0.2.0.exe",
        "installer_size": len(content),
        "installer_sha256": hashlib.sha256(content).hexdigest(),
        "release_notes_url": "https://github.com/example/release",
        "gpu_worker_protocol": 2,
    }


class UpdateManifestTests(unittest.TestCase):
    def test_signed_manifest_is_verified_and_parsed(self):
        document, public_key = signed_document(valid_payload())

        release = verify_signed_manifest(document, public_key)

        self.assertEqual(release.version, "0.2.0")
        self.assertEqual(release.installer_name, "ToonOut-Setup-0.2.0.exe")

    def test_tampered_manifest_is_rejected(self):
        payload = valid_payload()
        document, public_key = signed_document(payload)
        envelope = json.loads(document)
        envelope["payload"]["version"] = "9.9.9"

        with self.assertRaises(UpdateSecurityError):
            verify_signed_manifest(json.dumps(envelope).encode(), public_key)

    def test_empty_public_key_disables_updates(self):
        document, _ = signed_document(valid_payload())

        with self.assertRaises(UpdateConfigurationError):
            verify_signed_manifest(document, "")

    def test_version_comparison_is_numeric(self):
        self.assertTrue(is_newer_version("1.10.0", "1.9.9"))
        self.assertFalse(is_newer_version("1.9.9", "1.10.0"))


class UpdateDownloadTests(unittest.TestCase):
    def test_cache_cleanup_removes_only_old_version_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "0.1.0"
            current = root / "0.2.0"
            unrelated = root / "manual-backup"
            old.mkdir()
            current.mkdir()
            unrelated.mkdir()
            (old / "setup.exe").write_bytes(b"old")

            prune_update_cache(root, "0.2.0")

            self.assertFalse(old.exists())
            self.assertTrue(current.is_dir())
            self.assertTrue(unrelated.is_dir())

    def test_download_is_verified_before_final_name_is_exposed(self):
        content = b"verified installer bytes"
        payload = valid_payload(content)
        release = UpdateRelease(
            version=str(payload["version"]),
            minimum_supported_version=str(payload["minimum_supported_version"]),
            installer_url=str(payload["installer_url"]),
            installer_name=str(payload["installer_name"]),
            installer_size=int(payload["installer_size"]),
            installer_sha256=str(payload["installer_sha256"]),
            release_notes_url=str(payload["release_notes_url"]),
            gpu_worker_protocol=int(payload["gpu_worker_protocol"]),
        )
        progress = []

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "update_service.urlopen",
                return_value=FakeResponse(content),
            ),
        ):
            result = download_installer(
                release,
                directory,
                progress=lambda received, total: progress.append(
                    (received, total)
                ),
            )

            self.assertEqual(result.read_bytes(), content)
            self.assertFalse(
                (Path(directory) / f"{release.installer_name}.part").exists()
            )
            self.assertEqual(progress[-1], (len(content), len(content)))

    def test_corrupt_download_is_deleted(self):
        expected = b"expected"
        payload = valid_payload(expected)
        release = UpdateRelease(
            version=str(payload["version"]),
            minimum_supported_version=str(payload["minimum_supported_version"]),
            installer_url=str(payload["installer_url"]),
            installer_name=str(payload["installer_name"]),
            installer_size=int(payload["installer_size"]),
            installer_sha256=str(payload["installer_sha256"]),
            release_notes_url=str(payload["release_notes_url"]),
            gpu_worker_protocol=int(payload["gpu_worker_protocol"]),
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "update_service.urlopen",
                return_value=FakeResponse(b"corrupt!"),
            ),
        ):
            with self.assertRaises(UpdateSecurityError):
                download_installer(release, directory)

            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
