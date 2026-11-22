"""Verified storage archive creation and safe restore-to-new-directory CLI."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from app.core import persistence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@persistence.storage_locked
def create_backup(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Backup destination already exists: {destination}")
    persistence.check_storage()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sg-insight-backup-") as temp_dir:
        stage = Path(temp_dir)
        db_copy = stage / "sessions.db"
        source = sqlite3.connect(persistence.DB_PATH)
        target = sqlite3.connect(db_copy)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        files = {"sessions.db": _sha256(db_copy)}
        db = sqlite3.connect(db_copy)
        try:
            dataset_rows = db.execute("SELECT dataset_id, version_count FROM sessions").fetchall()
        finally:
            db.close()
        for dataset_id, version_count in dataset_rows:
            persistence._validated_dataset_id(dataset_id)
            for version in range(1, int(version_count) + 1):
                source_file = persistence._version_path(dataset_id, version)
                rel = f"versions/{dataset_id}/v{version}.pkl"
                staged = stage / rel
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, staged)
                files[rel] = _sha256(staged)
        manifest = {"format": 1, "files": files}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temp_output = Path(temp_name)
        try:
            with zipfile.ZipFile(temp_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(stage / "manifest.json", "manifest.json")
                for rel in files:
                    archive.write(stage / rel, rel)
            os.replace(temp_output, destination)
        finally:
            temp_output.unlink(missing_ok=True)
    return destination


def restore_backup(archive_path: Path, target_dir: Path) -> Path:
    target_dir = target_dir.resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError("Restore target must be empty or not yet created; active data is never overwritten.")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.namelist()
            infos = archive.infolist()
            total_unpacked = sum(info.file_size for info in infos)
            if len(entries) > 1_000_000 or total_unpacked > 10 * 1024 * 1024 * 1024 or len(entries) != len(set(entries)) or any(
                "\\" in info.filename or ":" in info.filename or PurePosixPath(info.filename).is_absolute()
                or ".." in PurePosixPath(info.filename).parts or stat.S_ISLNK(info.external_attr >> 16)
                for info in infos
            ):
                raise ValueError("Backup contains an unsafe file path.")
            if shutil.disk_usage(target_dir.parent).free < total_unpacked + persistence.MIN_FREE_DISK_BYTES:
                raise OSError("Not enough free space to restore this backup safely.")
            manifest_info = archive.getinfo("manifest.json")
            if manifest_info.file_size > 2 * 1024 * 1024:
                raise ValueError("Backup manifest exceeds the supported size.")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != 1 or not isinstance(manifest.get("files"), dict):
                raise ValueError("Unsupported backup format.")
            if set(entries) != set(manifest["files"]) | {"manifest.json"}:
                raise ValueError("Backup archive entries do not match its manifest.")
            extracted: list[Path] = []
            for rel, expected in manifest["files"].items():
                if rel not in entries or rel == "manifest.json":
                    raise ValueError("Backup manifest references a missing file.")
                safe_rel = PurePosixPath(rel)
                if safe_rel.is_absolute() or ".." in safe_rel.parts or not safe_rel.parts or not (
                    rel == "sessions.db" or re.fullmatch(r"versions/[0-9a-f-]{36}/v[1-9][0-9]*\.pkl", rel)
                ):
                    raise ValueError("Backup manifest contains an unsafe path.")
                path = target_dir.joinpath(*safe_rel.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(rel) as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(path)
                if _sha256(path) != expected:
                    raise ValueError(f"Backup checksum mismatch for {rel}.")
        db_path = target_dir / "sessions.db"
        db = sqlite3.connect(db_path)
        try:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Restored SQLite database failed its integrity check.")
            rows = db.execute("SELECT dataset_id, version_count FROM sessions").fetchall()
        finally:
            db.close()
        for dataset_id, count in rows:
            persistence._validated_dataset_id(dataset_id)
            for number in range(1, count + 1):
                if not (target_dir / "versions" / dataset_id / f"v{number}.pkl").is_file():
                    raise ValueError("Backup is missing an indexed dataset version.")
        return target_dir
    except Exception:
        # Keep partial recovery material for inspection; never touch the source archive or active storage.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create a verified backup archive")
    create.add_argument("destination", type=Path)
    restore = commands.add_parser("restore", help="Restore into an empty directory")
    restore.add_argument("archive", type=Path)
    restore.add_argument("target_dir", type=Path)
    args = parser.parse_args()
    path = create_backup(args.destination) if args.command == "create" else restore_backup(args.archive, args.target_dir)
    print(path)


if __name__ == "__main__":
    main()
