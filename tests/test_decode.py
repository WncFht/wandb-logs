import csv
import gzip
import importlib.util
import json
import math
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wandb_local", ROOT / "tools/wandb_local.py")
decoder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decoder)
pb = decoder.pb


def history(**values):
    record = pb.Record()
    for key, value in values.items():
        record.history.item.add(key=key, value_json=json.dumps(value))
    return record


class DecodeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.source = self.base / "run-test.wandb"
        self.output = self.base / "derived"

    def write(self, records):
        self.source.unlink(missing_ok=True)
        writer = decoder.DataStore()
        writer.open_for_write(str(self.source))
        try:
            for record in records:
                writer.write(record)
        finally:
            writer.close()

    def decode(self):
        with patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")):
            return decoder.decode_file(self.source, self.output)

    def test_sparse_history_special_values_and_unsampled_system_records(self):
        system = pb.Record()
        system.stats.timestamp.seconds = 100
        system.stats.item.add(key="gpu.0.utilization", value_json="0")
        exit_record = pb.Record()
        exit_record.exit.exit_code = 0
        self.write(
            [
                history(_step=0, loss=0, flag=False, text="中文", special=float("nan"), large=2**60),
                history(_step=5, accuracy=0.75, media={"_type": "image-file", "path": "media/a.png"}),
                system,
                system,
                exit_record,
            ]
        )
        original = self.source.read_bytes()
        self.source.chmod(0o444)
        metadata, cached = self.decode()
        rows = [json.loads(line) for line in (self.output / "history.jsonl").read_text().splitlines()]
        self.assertFalse(cached)
        self.assertEqual(metadata["status"], "finished")
        self.assertEqual(metadata["step_range"], [0, 5])
        self.assertEqual(rows[0]["large"], 2**60)
        self.assertIs(rows[0]["flag"], False)
        self.assertTrue(math.isnan(rows[0]["special"]))
        self.assertNotIn("loss", rows[1])
        self.assertEqual(rows[1]["media"]["path"], "media/a.png")
        with gzip.open(self.output / "system.jsonl.gz", "rt") as stream:
            stats = [json.loads(line) for line in stream]
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0]["metrics"]["gpu.0.utilization"], 0)
        self.assertEqual(original, self.source.read_bytes())
        self.assertTrue(self.decode()[1])
        with decoder.export_csv(self.output).open() as stream:
            exported = list(csv.DictReader(stream))
        self.assertEqual(exported[1]["loss"], "")

    def test_ordered_config_summary_updates_and_literal_keys(self):
        run = pb.Record()
        run.run.run_id = "test"
        run.run.config.update.add(key="optim", value_json='{"lr": 1, "keep": 2}')
        run.run.config.update.add(key="optim.lr", value_json="7")
        config = pb.Record()
        config.config.update.add(nested_key=["optim", "lr"], value_json="0.1")
        config.config.remove.add(nested_key=["optim", "keep"])
        first = pb.Record()
        first.summary.update.add(key="score", value_json="1")
        second = pb.Record()
        second.summary.remove.add(key="score")
        second.summary.update.add(key="train/loss", value_json="0.2")
        nested = history(**{"a.b": 3, "a/b": 4})
        nested.history.item.add(nested_key=["a", "b"], value_json="5")
        self.write([run, config, first, second, nested])
        metadata, _ = self.decode()
        self.assertEqual(metadata["status"], "snapshot")
        self.assertEqual(json.loads((self.output / "config.json").read_text()), {"optim": {"lr": 0.1}, "optim.lr": 7})
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), {"train/loss": 0.2})
        row = json.loads((self.output / "history.jsonl").read_text())
        self.assertEqual(row, {"a.b": 3, "a/b": 4, "a": {"b": 5}})

    def test_environment_records_preserve_partial_updates_and_writer_identity(self):
        first, second = pb.Record(), pb.Record()
        first.environment.python = "CPython 3.12"
        first.environment.writer_id = "driver"
        second.environment.gpu_count = 8
        second.environment.gpu_type = "NVIDIA A800"
        second.environment.writer_id = "driver"
        self.write([first, history(_step=1), second])
        self.decode()
        rows = [json.loads(x) for x in (self.output / "environment.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["python"], "CPython 3.12")
        self.assertNotIn("python", rows[1])
        self.assertEqual(rows[1]["gpu_count"], 8)
        self.assertEqual(rows[1]["writer_id"], "driver")

    def test_truncated_fragment_preserves_prefix_and_reports_error(self):
        self.write([history(_step=1, loss=0.5), history(_step=2, text="x" * 80000)])
        self.source.write_bytes(self.source.read_bytes()[:32768])
        metadata, _ = self.decode()
        self.assertEqual(metadata["status"], "partial_snapshot")
        self.assertEqual(metadata["history_rows"], 1)
        self.assertIsNotNone(metadata["error"])
        self.assertEqual(len((self.output / "history.jsonl").read_text().splitlines()), 1)

    def test_bad_header_and_missing_value_are_not_silently_accepted(self):
        self.source.write_bytes(b"invalid")
        self.assertEqual(self.decode()[0]["status"], "decode_error")
        broken = pb.Record()
        broken.history.item.add(key="missing")
        self.write([history(_step=1), broken])
        metadata, _ = self.decode()
        self.assertEqual(metadata["history_rows"], 1)
        self.assertEqual(metadata["status"], "decode_error")

    def test_corrupt_record_is_not_treated_as_an_unfinished_snapshot(self):
        self.write([history(_step=1), history(_step=2, loss=0.5)])
        data = bytearray(self.source.read_bytes())
        data[-1] ^= 1
        self.source.write_bytes(data)
        metadata, _ = self.decode()
        self.assertEqual(metadata["status"], "decode_error")
        self.assertEqual(metadata["history_rows"], 1)

    def test_changed_source_invalidates_cached_output(self):
        self.write([history(_step=1)])
        self.decode()
        self.write([history(_step=1), history(_step=2)])
        metadata, cached = self.decode()
        self.assertFalse(cached)
        self.assertEqual(metadata["history_rows"], 2)
        with self.assertRaisesRegex(ValueError, "original log"):
            decoder.decode_file(self.source, self.source.parent)

    def test_discovery_does_not_duplicate_latest_run_symlink(self):
        run = self.base / "project/wandb/offline-run-test"
        run.mkdir(parents=True)
        (run / "run-test.wandb").write_bytes(b"placeholder")
        (run.parent / "latest-run").symlink_to(run, target_is_directory=True)
        with patch.object(decoder, "ROOT", self.base):
            self.assertEqual(decoder.discover("project"), [run / "run-test.wandb"])


if __name__ == "__main__":
    unittest.main()
