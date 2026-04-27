import logging

import fsspec

from torch_dataloader_utils.splits.core import DataFileInfo

logger = logging.getLogger(__name__)

_BACKEND_INSTALL = {
    "s3": "pip install torch-dataloader-utils[s3]",
    "s3a": "pip install torch-dataloader-utils[s3]",
    "gs": "pip install torch-dataloader-utils[gcs]",
    "gcs": "pip install torch-dataloader-utils[gcs]",
    "az": "pip install torch-dataloader-utils[azure]",
    "abfs": "pip install torch-dataloader-utils[azure]",
}


def _install_hint(path: str) -> str:
    scheme = path.split("://")[0] if "://" in path else "file"
    return _BACKEND_INSTALL.get(scheme, "pip install torch-dataloader-utils")


def _raise_clean_fs_error(exc: Exception, path: str) -> None:
    """Re-raise *exc* as a more actionable error type based on its message.

    Inspects the exception message for known keywords so that no direct
    dependency on boto3/gcsfs/adlfs is required in the detection logic.
    """
    msg = str(exc).lower()

    credential_keywords = (
        "credential",
        "no credentials",
        "accessdenied",
        "autherror",
        "authenticationerror",
        "could not connect",
        "no auth",
        "unauthorized",
    )
    forbidden_keywords = (
        "403",
        "forbidden",
        "permission denied",
        "access denied",
        "accessforbidden",
        "not authorized",
    )
    notfound_keywords = (
        "404",
        "not found",
        "nosuchbucket",
        "nosuchkey",
        "no such file",
        "does not exist",
        "entitynotfound",
    )
    timeout_keywords = ("timeout", "timed out", "connection timed out")
    ssl_keywords = ("ssl", "certificate", "tls", "handshake")

    if any(k in msg for k in credential_keywords):
        raise PermissionError(
            f"No credentials found for path {path!r} — "
            "set environment variables (e.g. AWS_ACCESS_KEY_ID) or pass storage_options"
        ) from exc
    if any(k in msg for k in forbidden_keywords):
        raise PermissionError(
            f"Access denied for path {path!r} — check IAM roles or bucket policy"
        ) from exc
    if any(k in msg for k in notfound_keywords):
        raise FileNotFoundError(f"Path not found: {path}") from exc
    if any(k in msg for k in timeout_keywords):
        raise TimeoutError(f"Connection timed out for path {path!r}") from exc
    if any(k in msg for k in ssl_keywords):
        raise OSError(f"SSL error connecting to {path!r}") from exc

    # Unknown error — re-raise unchanged
    raise


def discover_files(
    path: str,
    storage_options: dict | None = None,
    extensions: list[str] | None = None,
) -> list[DataFileInfo]:
    """Discover files at *path* and return a list of DataFileInfo.

    Args:
        path: A directory URI, glob pattern, or single file URI.
        storage_options: Passed directly to fsspec (credentials, endpoint, etc.).
        extensions: Optional list of extensions to keep, e.g. [".parquet", ".orc"].
                    When None all files are returned.

    Returns:
        list[DataFileInfo] with path and file_size populated.

    Raises:
        FileNotFoundError: When the path does not exist.
        PermissionError: When credentials are missing or access is denied.
        TimeoutError: When the connection times out.
        OSError: When an SSL/TLS error occurs.
        ImportError: When a required optional backend is not installed.
    """
    opts = storage_options or {}

    logger.info("Discovering files: path=%s", path)

    try:
        fs, resolved = fsspec.url_to_fs(path, **opts)
    except ImportError as exc:
        hint = _install_hint(path)
        raise ImportError(f"{exc}\n\nInstall the required backend: {hint}") from exc

    logger.debug("Filesystem backend: %s  resolved=%s", type(fs).__name__, resolved)

    # --- determine path type and collect raw stat entries ---
    is_glob = "*" in path or "?" in path

    try:
        if is_glob:
            matched = fs.glob(resolved)
            logger.debug("Glob matched %d path(s) for pattern %s", len(matched), resolved)
            # glob returns paths — collect stat for each
            stats = [fs.stat(p) for p in matched if fs.isfile(p)]
        elif fs.exists(resolved):
            if fs.isdir(resolved):
                entries = fs.ls(resolved, detail=True)
                stats = [e for e in entries if e.get("type", "").lower() == "file"]
                logger.debug("Directory listing: %d entries, %d file(s)", len(entries), len(stats))
            else:
                stats = [fs.stat(resolved)]
                logger.debug("Single file path detected")
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    except (FileNotFoundError, ImportError):
        raise
    except Exception as exc:
        _raise_clean_fs_error(exc, path)

    # --- determine scheme prefix to restore on fsspec-internal paths ---
    # fsspec strips the scheme from paths returned in stat/ls/glob results.
    # e.g. s3fs returns "bucket/key" instead of "s3://bucket/key".
    # Re-attach the scheme so reader.py can reconstruct the correct filesystem.
    scheme = path.split("://")[0] if "://" in path else None
    is_local = fs.protocol in ("file", "local") or (
        isinstance(fs.protocol, tuple) and "file" in fs.protocol
    )

    def _restore_path(raw: str) -> str:
        if is_local or scheme is None:
            return raw
        # Only add the scheme if not already present
        if "://" in raw:
            return raw
        return f"{scheme}://{raw}"

    # --- build DataFileInfo list ---
    files: list[DataFileInfo] = []
    for stat in stats:
        file_path = _restore_path(stat.get("name") or stat.get("path", ""))
        file_size = stat.get("size")

        if extensions is not None:
            if not any(file_path.endswith(ext) for ext in extensions):
                logger.debug("Skipping (extension filter): %s", file_path)
                continue

        info = DataFileInfo(path=file_path, file_size=file_size)
        logger.debug("Found: %s  size=%s bytes", file_path, file_size)
        files.append(info)

    logger.info(
        "Discovery complete: %d file(s) found at %s  total_size=%s bytes",
        len(files),
        path,
        sum(f.file_size for f in files if f.file_size is not None) or "unknown",
    )
    return files
