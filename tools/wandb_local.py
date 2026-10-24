#!/usr/bin/env python3
"""Decode local W&B transaction logs; never initialize, sync or contact a W&B API."""

import argparse
import csv
import gzip
import io
import json
import shutil
import struct
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import wandb
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[1]
DECODER_VERSION = 2


class SnapshotStore(DataStore):
    """Use the SDK framing reader over an in-memory snapshot, without its r+b open."""

    def __init__(self, data):
        super().__init__()
        self._fname = "<read-only snapshot>"
        self._fp = io.BytesIO(data)
        self._index = 0
        self._size_bytes = len(data)
        self._opened_for_scan = True
        try:
            self._read_header()
        except Exception as exc:
            self._fp.close()
            raise ValueError("Invalid or truncated W&B file header") from exc


def json_value(item):
    """Prefer newer typed values; support the JSON encoding used by SDK 0.28.1."""
    if "value" in item.DESCRIPTOR.oneofs_by_name:
        kind = item.WhichOneof("value")
        if kind:
            if kind == "none":
                return None
            value = getattr(item, kind)
            return json.loads(value) if kind == "json" else value
    if item.value_json:
        return json.loads(item.value_json)
    raise ValueError("Record value is missing; a newer W&B protobuf schema may be required")


def key_path(item):
    """Keep literal slash/dot metric names distinct from protobuf nested paths."""
    if item.key:
        return [item.key]
    if getattr(item, "nested_key", None):
        return list(item.nested_key)
    raise ValueError("Record item has no key")


def apply_updates(target, updates, removals=()):
    for item in updates:
        path = key_path(item)
        node = target
        for key in path[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[path[-1]] = json_value(item)
    for item in removals:
        path = key_path(item)
        node = target
        for key in path[:-1]:
            node = node.get(key)
            if not isinstance(node, dict):
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)


def protobuf_dict(message):
    return MessageToDict(message, preserving_proto_field_name=True)


def write_row(stream, value):
    # Retain NaN/Infinity as supported by Python JSON and W&B's own JSON logs.
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=True) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def source_stat(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def decode_file(source, destination, force=False):
    """Export valid records and report missing exits or malformed tails explicitly."""
    source = Path(source)
    destination = Path(destination)
    if source.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Decoder output must not contain the original log")
    stamp = source_stat(source)
    cache_key = {"source": str(source.resolve()), **stamp, "decoder": DECODER_VERSION, "reader": wandb.__version__}
    metadata_path = destination / "run.json"
    expected = (
        "history.jsonl",
        "system.jsonl.gz",
        "config.json",
        "summary.json",
        "console.jsonl",
        "metric-definitions.jsonl",
        "artifacts.jsonl",
        "environment.jsonl",
    )
    if not force and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            previous.get("cache_key") == cache_key
            and previous.get("status") in ("finished", "snapshot", "partial_snapshot")
            and all((destination / name).is_file() for name in expected)
        ):
            return previous, True

    # Read the shared filesystem once; SDK scans then operate on local memory.
    data = source.read_bytes()
    changed = source_stat(source) != stamp or len(data) != stamp["bytes"]
    counts = Counter()
    written = Counter()
    config, summary, run_info = {}, {}, {}
    steps = []
    metric_keys = set()
    exit_code = None
    error = None
    offset = 0
    header = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".decode-", dir=destination.parent) as temp:
        output = Path(temp)
        with ExitStack() as stack:
            streams = {
                name: stack.enter_context((output / name).open("w", encoding="utf-8"))
                for name in expected
                if name not in ("system.jsonl.gz", "config.json", "summary.json")
            }
            system = stack.enter_context(gzip.open(output / "system.jsonl.gz", "wt", encoding="utf-8"))
            reader = None
            try:
                reader = SnapshotStore(data)
                while True:
                    offset = reader.get_offset()
                    raw = reader.scan_data()
                    if raw is None:
                        tail = data[offset:]
                        legal_padding = len(tail) < 7 and not tail.strip(b"\0") and len(data) % 32768 == 0
                        if tail and not legal_padding:
                            raise EOFError("EOF inside a fragmented record")
                        break
                    record = pb.Record()
                    record.ParseFromString(raw)
                    kind = record.WhichOneof("record_type")
                    if kind is None:
                        raise ValueError("Unknown record type; upgrade the W&B protobuf schema")
                    counts[kind] += 1
                    if kind == "header":
                        header = protobuf_dict(record.header)
                    elif kind == "run":
                        run_info.update(protobuf_dict(record.run))
                        run_info.pop("config", None)
                        run_info.pop("summary", None)
                        apply_updates(config, record.run.config.update, record.run.config.remove)
                        apply_updates(summary, record.run.summary.update, record.run.summary.remove)
                    elif kind in ("config", "summary"):
                        changes = getattr(record, kind)
                        apply_updates(config if kind == "config" else summary, changes.update, changes.remove)
                    elif kind == "history":
                        row = {}
                        apply_updates(row, record.history.item)
                        if "_step" not in row and record.history.HasField("step"):
                            row["_step"] = record.history.step.num
                        if isinstance(row.get("_step"), (int, float)):
                            steps.append(row["_step"])
                        metric_keys.update(row)
                        write_row(streams["history.jsonl"], row)
                        written["history"] += 1
                    elif kind == "stats":
                        row = {}
                        apply_updates(row, record.stats.item)
                        write_row(system, {"timestamp": record.stats.timestamp.ToJsonString(), "metrics": row})
                        written["stats"] += 1
                    elif kind in ("output", "output_raw", "output_logger"):
                        write_row(streams["console.jsonl"], {"type": kind, **protobuf_dict(getattr(record, kind))})
                    elif kind == "metric":
                        write_row(streams["metric-definitions.jsonl"], protobuf_dict(record.metric))
                    elif kind == "artifact":
                        write_row(streams["artifacts.jsonl"], protobuf_dict(record.artifact))
                    elif kind == "environment":
                        write_row(streams["environment.jsonl"], protobuf_dict(record.environment))
                    elif kind == "exit":
                        exit_code = record.exit.exit_code
            except (AssertionError, DecodeError, EOFError, IndexError, KeyError, ValueError, struct.error) as exc:
                error = {"offset": offset, "type": type(exc).__name__, "message": str(exc)}
            finally:
                if reader is not None:
                    reader.close()
        if error:
            status = "partial_snapshot" if error["type"] == "EOFError" else "decode_error"
        else:
            status = "finished" if exit_code is not None else "snapshot"
        if changed:
            status = "source_changed"
        metadata = {
            "cache_key": cache_key,
            "kind": "wandb-offline",
            "status": status,
            "exit_code": exit_code,
            "run": run_info,
            "header": header,
            "record_counts": dict(counts),
            "history_rows": written["history"],
            "system_rows": written["stats"],
            "step_range": [min(steps), max(steps)] if steps else None,
            "history_keys": sorted(metric_keys),
            "error": error,
            "note": "snapshot means no exit record; it does not establish whether the run is running or crashed",
        }
        write_json(output / "config.json", config)
        write_json(output / "summary.json", summary)
        write_json(output / "run.json", metadata)
        # This directory contains only reproducible decoder output, never original logs.
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(output), destination)
    return metadata, False


def discover(project, selector=None):
    if Path(project).name != project or project in (".", ".."):
        raise ValueError("Project must be one directory name")
    paths = sorted(
        p
        for p in (ROOT / project / "wandb").glob("*/run-*.wandb")
        if not p.parent.is_symlink() and not p.is_symlink() and p.parent.name.startswith(("offline-run-", "run-"))
    )
    if selector:
        paths = [p for p in paths if p.parent.name == selector or p.stem == "run-" + selector]
    if not paths:
        raise ValueError("No matching local .wandb files")
    return paths


def output_dir(project, source):
    return ROOT / "derived" / project / source.parent.name


def export_csv(directory):
    with (directory / "history.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    keys = sorted({key for row in rows for key in row})
    if "_step" in keys:
        keys.remove("_step")
        keys.insert(0, "_step")
    output = directory / "history.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "decode", "csv"))
    parser.add_argument("--project", default="MirrorOPD-verl")
    parser.add_argument("--run", help="run ID or full run directory name")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        paths = discover(args.project, args.run)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "csv" and len(paths) != 1:
        parser.error("CSV export requires exactly one run; use --run")
    failed = False
    for source in paths:
        output = output_dir(args.project, source)
        if args.command == "list":
            metadata = json.loads((output / "run.json").read_text()) if (output / "run.json").is_file() else None
            status = metadata["status"] if metadata else "not_decoded"
            if metadata and {
                "bytes": metadata["cache_key"]["bytes"],
                "mtime_ns": metadata["cache_key"]["mtime_ns"],
            } != source_stat(source):
                status = "stale"
            print(f"{source.parent.name}\t{source.stat().st_size / 1048576:.2f} MiB\t{status}", flush=True)
            continue
        metadata, cached = decode_file(source, output, args.force)
        status = metadata["status"]
        print(
            f"{source.parent.name}: {status}, history={metadata['history_rows']}, system={metadata['system_rows']}"
            f"{' (cached)' if cached else ''}",
            flush=True,
        )
        if status in ("decode_error", "source_changed"):
            failed = True
            print(f"  Partial output: {output}; details in run.json", flush=True)
        else:
            if status == "partial_snapshot":
                print(
                    "  Exported complete records only; the unfinished tail record is retained in the original log",
                    flush=True,
                )
            if args.command == "csv":
                print(export_csv(output), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
