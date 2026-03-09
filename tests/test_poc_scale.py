"""
Tests for poc_scale.py — inference scaling proof-of-concept.

All tests run on CPU with tiny synthetic inputs (no GPU or checkpoint required).
"""

from __future__ import annotations

import csv
import sys
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest
import torch

# Add repo root to path so we can import poc_scale directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import poc_scale as ps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


def _small_result(size: int = 64, status: str = "OK") -> ps.BenchResult:
    if status == "OOM":
        return ps.BenchResult(size=size, vram_gb=0.0, mean_ms=0.0, p95_ms=0.0, fps=0.0, speedup=0.0, status="OOM")
    return ps.BenchResult(
        size=size,
        vram_gb=1.5,
        mean_ms=100.0,
        p95_ms=110.0,
        fps=10.0,
        speedup=1.0,
        status="OK",
        timings_ms=[95.0, 100.0, 105.0],
    )


# ---------------------------------------------------------------------------
# Unit tests for run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """run_benchmark on CPU with tiny sizes — no GPU required."""

    def test_returns_one_result_per_size(self):
        results = ps.run_benchmark(sizes=[32, 64], device=CPU, warmup=1, runs=2, use_refiner=False, use_amp=False)
        assert len(results) == 2

    def test_sizes_match_requested(self):
        results = ps.run_benchmark(sizes=[32, 64], device=CPU, warmup=1, runs=2, use_refiner=False, use_amp=False)
        assert [r.size for r in results] == [32, 64]

    def test_all_ok_on_cpu(self):
        results = ps.run_benchmark(sizes=[32], device=CPU, warmup=1, runs=2, use_refiner=False, use_amp=False)
        assert results[0].status == "OK"

    def test_timings_populated(self):
        results = ps.run_benchmark(sizes=[32], device=CPU, warmup=1, runs=3, use_refiner=False, use_amp=False)
        assert len(results[0].timings_ms) == 3

    def test_fps_positive(self):
        results = ps.run_benchmark(sizes=[32], device=CPU, warmup=1, runs=2, use_refiner=False, use_amp=False)
        assert results[0].fps > 0

    def test_mean_ms_positive(self):
        results = ps.run_benchmark(sizes=[32], device=CPU, warmup=1, runs=2, use_refiner=False, use_amp=False)
        assert results[0].mean_ms > 0

    def test_speedup_baseline_is_1(self):
        """The first OK result should have speedup == 1.0 (it is the baseline)."""
        results = ps.run_benchmark(sizes=[32, 64], device=CPU, warmup=1, runs=2, use_refiner=False, use_amp=False)
        ok = [r for r in results if r.status == "OK"]
        assert ok[0].speedup == pytest.approx(1.0)

    def test_smaller_size_faster_or_equal(self):
        """Smaller input should never be slower than a larger one (on average)."""
        results = ps.run_benchmark(sizes=[32, 64], device=CPU, warmup=1, runs=3, use_refiner=False, use_amp=False)
        ok = [r for r in results if r.status == "OK"]
        if len(ok) == 2:
            assert ok[0].mean_ms <= ok[1].mean_ms * 2  # allow 2× slack for CI noise

    def test_vram_zero_on_cpu(self):
        results = ps.run_benchmark(sizes=[32], device=CPU, warmup=0, runs=1, use_refiner=False, use_amp=False)
        assert results[0].vram_gb == 0.0


# ---------------------------------------------------------------------------
# OOM simulation
# ---------------------------------------------------------------------------


class TestOOMHandling:
    def test_oom_during_build_returns_oom_status(self):
        """If the model build raises OOM, the result status must be OOM."""
        oom = torch.cuda.OutOfMemoryError("simulated OOM")
        with mock.patch("poc_scale._build_model", side_effect=oom):
            results = ps.run_benchmark(sizes=[64], device=CPU, warmup=1, runs=2)
        assert results[0].status == "OOM"

    def test_oom_during_inference_returns_oom_status(self):
        """OOM during inference (not build) must also return OOM status."""

        call_count = [0]
        real_build = ps._build_model

        def patched_build(img_size, use_refiner, device):
            model = real_build(img_size, use_refiner, device)
            # Wrap forward to raise OOM on first call
            original_forward = model.forward

            def boom_forward(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise torch.cuda.OutOfMemoryError("simulated OOM in forward")
                return original_forward(*args, **kwargs)

            model.forward = boom_forward
            return model

        with mock.patch("poc_scale._build_model", side_effect=patched_build):
            results = ps.run_benchmark(sizes=[32], device=CPU, warmup=1, runs=2, use_refiner=False, use_amp=False)

        assert results[0].status == "OOM"


# ---------------------------------------------------------------------------
# print_table output
# ---------------------------------------------------------------------------


class TestPrintTable:
    def test_prints_without_error(self, capsys):
        results = [_small_result(64), _small_result(128)]
        ps.print_table(results)
        captured = capsys.readouterr()
        assert "size" in captured.out
        assert "vram(GB)" in captured.out

    def test_oom_row_shows_dash(self, capsys):
        results = [_small_result(32, status="OOM")]
        ps.print_table(results)
        captured = capsys.readouterr()
        assert "OOM" in captured.out
        assert "—" in captured.out

    def test_ok_row_shows_numbers(self, capsys):
        results = [_small_result(64)]
        ps.print_table(results)
        captured = capsys.readouterr()
        assert "64" in captured.out
        assert "1.50" in captured.out  # vram_gb


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_csv_contains_headers(self, tmp_path):
        out = tmp_path / "out.csv"
        ps.write_csv([_small_result(64)], str(out))
        content = out.read_text()
        reader = csv.DictReader(StringIO(content))
        assert "size" in reader.fieldnames
        assert "vram_gb" in reader.fieldnames
        assert "mean_ms" in reader.fieldnames
        assert "fps" in reader.fieldnames
        assert "status" in reader.fieldnames

    def test_csv_row_count(self, tmp_path):
        out = tmp_path / "out.csv"
        ps.write_csv([_small_result(64), _small_result(128)], str(out))
        content = out.read_text()
        reader = list(csv.DictReader(StringIO(content)))
        assert len(reader) == 2

    def test_csv_does_not_contain_timings_ms(self, tmp_path):
        out = tmp_path / "out.csv"
        ps.write_csv([_small_result(64)], str(out))
        content = out.read_text()
        assert "timings_ms" not in content


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_defaults(self):
        args = ps._parse_args([])
        assert args.device == "auto"
        assert args.warmup == 2
        assert args.runs == 5
        assert not args.no_refiner
        assert not args.no_amp
        assert 2048 in args.sizes

    def test_custom_sizes(self):
        args = ps._parse_args(["--sizes", "256", "512"])
        assert args.sizes == [256, 512]

    def test_device_flag(self):
        args = ps._parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_no_refiner_flag(self):
        args = ps._parse_args(["--no-refiner"])
        assert args.no_refiner is True

    def test_output_flag(self):
        args = ps._parse_args(["--output", "results.csv"])
        assert args.output == "results.csv"

    def test_warmup_runs(self):
        args = ps._parse_args(["--warmup", "3", "--runs", "10"])
        assert args.warmup == 3
        assert args.runs == 10


# ---------------------------------------------------------------------------
# End-to-end main() on CPU
# ---------------------------------------------------------------------------


class TestMainCli:
    def test_main_returns_zero(self):
        rc = ps.main(["--sizes", "32", "--device", "cpu", "--warmup", "1", "--runs", "2", "--no-refiner"])
        assert rc == 0

    def test_main_writes_csv(self, tmp_path):
        out = str(tmp_path / "scale.csv")
        ps.main(["--sizes", "32", "--device", "cpu", "--warmup", "1", "--runs", "2", "--no-refiner", "--output", out])
        content = Path(out).read_text()
        assert "size" in content
        assert "OK" in content

    def test_main_prints_table(self, capsys):
        ps.main(["--sizes", "32", "--device", "cpu", "--warmup", "1", "--runs", "2", "--no-refiner"])
        captured = capsys.readouterr()
        assert "fps" in captured.out
        assert "32" in captured.out
