# Model storage location

## Problem

Hugging Face normally uses a user cache directory. That directory may be inaccessible, and a large model on the system drive is inconvenient for some users.

## Options considered

- Rely on the global Hugging Face cache.
- Bundle model files inside the application installer.
- Use an app-specific default and let the user choose another folder.

## Decision

Use `%LOCALAPPDATA%\ToonOut\models` by default. Show actual on-disk installation state in the main header. Before the first download, show the location and allow the user to choose a writable folder. Persist the choice with `QSettings` and pass it explicitly as the app-owned Hugging Face download location.

## Consequences

- Model downloads no longer depend on the global `.cache` directory.
- Model snapshots contain regular files rather than symbolic links; see decision 008.
- The base BiRefNet weights are not downloaded. The app builds the pinned architecture from config and directly assigns the full ToonOut state dict, saving about 424 MiB and avoiding float16/float32 mismatch.
- Users can place large files on another drive.
- Moving a model copies and validates the new cache before deleting the old cache.
- Deletion targets only the two managed model repositories and their small lock/module cache entries, never the selected root folder.
- The default location needs no elevation. A protected user-selected location can trigger a Windows UAC restart, preserving queued image paths.
