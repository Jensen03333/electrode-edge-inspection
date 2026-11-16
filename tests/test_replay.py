import tempfile
import unittest
from pathlib import Path

from station_replay import image_names, load_timings, simulate, summarize


class ReplayTests(unittest.TestCase):
    def test_bounded_queue_drops_new_arrivals_and_reports_late_completions(self):
        trace = simulate(["a", "b"], {"a": 35, "b": 10}, 10, 1, 30, cycles=3)
        self.assertEqual(
            [row["status"] for row in trace],
            ["late", "late", "dropped_queue_full", "dropped_queue_full", "late", "late"],
        )
        self.assertIsNone(trace[2]["start_ms"])
        self.assertEqual(trace[4]["queue_wait_ms"], 5)
        summary = summarize(trace)
        self.assertEqual(summary["completed"], 4)
        self.assertEqual(summary["completed_late"], 4)
        self.assertEqual(summary["dropped_queue_full"], 2)
        self.assertEqual(summary["elapsed_simulated_ms"], 90)
        self.assertEqual(summary["completed_latency_ms"]["max"], 40)

    def test_finishing_on_arrival_or_deadline_is_on_time(self):
        trace = simulate(["a"], {"a": 0.1}, 0.1, 0, 0.1, cycles=7)
        self.assertTrue(all(row["status"] == "on_time" for row in trace))
        self.assertAlmostEqual(summarize(trace)["completed_latency_ms"]["p95"], 0.1)

    def test_idle_worker_starts_at_arrival(self):
        trace = simulate(["a"], {"a": 3}, 20, 0, 5, cycles=3)
        self.assertEqual([row["start_ms"] for row in trace], [0, 20, 40])
        self.assertEqual(summarize(trace)["elapsed_simulated_ms"], 43)

    def test_no_waiting_slots_drop_while_worker_busy(self):
        trace = simulate(["a"], {"a": 21}, 10, 0, 25, cycles=4)
        self.assertEqual([row["status"] for row in trace], [
            "on_time", "dropped_queue_full", "dropped_queue_full", "on_time",
        ])

    def test_timing_csv_matches_nested_relative_paths_and_flags_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "batch").mkdir()
            (root / "batch" / "frame.JPG").touch()
            names = image_names(root)
            self.assertEqual(names, ["batch/frame.JPG"])
            timings = root / "service.csv"
            timings.write_text("file,service_ms\nbatch/frame.JPG,23.5\n", encoding="utf-8")
            self.assertEqual(load_timings(timings, names), {"batch/frame.JPG": 23.5})
            with self.assertRaisesRegex(ValueError, "Missing timings"):
                load_timings(timings, ["new.jpg"])
            timings.write_text("file,service_ms\na.jpg,NaN\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_timings(timings, ["a.jpg"])


if __name__ == "__main__":
    unittest.main()
