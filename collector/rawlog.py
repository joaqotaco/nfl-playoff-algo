"""Append-only raw storage: one gzipped JSONL file per round.

A round is written to a plain .jsonl file, flushed line by line, then compressed
once finished. A crash therefore loses nothing already written: the next run
compresses whatever was left behind. Finished files are never modified.
"""
import gzip
import json
import os
import shutil
from pathlib import Path


class RoundLog:
    def __init__(self, raw_dir, round_id):
        day_dir = Path(raw_dir) / round_id[:4] / round_id[4:6] / round_id[6:8]
        day_dir.mkdir(parents=True, exist_ok=True)
        self.path = day_dir / f"{round_id}.jsonl"
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, record):
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
        compress(self.path)


def compress(path):
    path = Path(path)
    target = path.with_suffix(".jsonl.gz")
    tmp = path.with_suffix(".jsonl.gz.tmp")
    with open(path, "rb") as source, gzip.open(tmp, "wb") as sink:
        shutil.copyfileobj(source, sink)
    os.replace(tmp, target)
    path.unlink()


def recover(raw_dir):
    """Compress files left by a crashed round. Call while holding the run lock."""
    leftovers = sorted(Path(raw_dir).rglob("*.jsonl"))
    for path in leftovers:
        compress(path)
    return leftovers


def read_records(path):
    """Yield records from a raw file. A crash can truncate only the final line."""
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                return
