"""Replay image arrivals through a single FIFO inspection worker in simulated time."""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TRACE_FIELDS = (
    "frame_id", "file", "arrival_ms", "start_ms", "finish_ms", "service_ms",
    "queue_wait_ms", "latency_ms", "status",
)


def positive_ms(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Expected a positive, finite duration in ms, got {value!r}")
    return number


def image_names(directory):
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    names = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not names:
        raise ValueError(f"No JPG or PNG images found in {directory}")
    return names


def load_timings(path, names):
    """Read per-frame worker durations keyed by image-root-relative path."""
    durations = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"file", "service_ms"}.issubset(reader.fieldnames or []):
            raise ValueError("Timing CSV requires the columns file,service_ms")
        for line, row in enumerate(reader, start=2):
            name = (row["file"] or "").strip().replace("\\", "/")
            if not name or name in durations:
                raise ValueError(f"Empty or duplicate timing file on CSV line {line}: {name!r}")
            durations[name] = positive_ms(row["service_ms"] or "")
    missing = [name for name in names if name not in durations]
    if missing:
        raise ValueError(f"Missing timings for {len(missing)} input(s), first: {missing[0]}")
    return durations


def simulate(names, durations, arrival_ms, queue_capacity, deadline_ms, cycles=1):
    """Queue capacity counts waiting frames, in addition to one active worker."""
    arrival_ms = positive_ms(arrival_ms)
    deadline_ms = positive_ms(deadline_ms)
    if queue_capacity < 0 or cycles < 1 or not names:
        raise ValueError("Supply input frames, queue_capacity >= 0 and cycles >= 1")
    durations = {name: positive_ms(durations[name]) for name in names}
    unfinished = deque()
    trace = []
    for frame_id in range(cycles * len(names)):
        name = names[frame_id % len(names)]
        arrival = frame_id * arrival_ms
        # A completion exactly at arrival frees the worker before admission.
        while unfinished and unfinished[0] <= arrival + 1e-9:
            unfinished.popleft()
        row = dict.fromkeys(TRACE_FIELDS)
        row.update(frame_id=frame_id, file=name, arrival_ms=arrival)
        if len(unfinished) >= queue_capacity + 1:
            row["status"] = "dropped_queue_full"
        else:
            start = max(arrival, unfinished[-1]) if unfinished else arrival
            finish = start + durations[name]
            unfinished.append(finish)
            latency = finish - arrival
            row.update(
                start_ms=start, finish_ms=finish, service_ms=durations[name],
                queue_wait_ms=start - arrival, latency_ms=latency,
                status="on_time" if latency <= deadline_ms + 1e-9 else "late",
            )
        trace.append(row)
    return trace


def percentile(sorted_values, fraction):
    index = (len(sorted_values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def summarize(trace):
    completed = [row for row in trace if row["finish_ms"] is not None]
    latencies = sorted(row["latency_ms"] for row in completed)
    elapsed_ms = max(trace[-1]["arrival_ms"], completed[-1]["finish_ms"])
    late = sum(row["status"] == "late" for row in completed)
    return {
        "arrivals": len(trace),
        "completed": len(completed),
        "dropped_queue_full": len(trace) - len(completed),
        "completed_late": late,
        "delivered_on_time": len(completed) - late,
        "elapsed_simulated_ms": elapsed_ms,
        "completed_frames_per_second": 1000 * len(completed) / elapsed_ms,
        "completed_latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": latencies[-1],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Image-only directory, searched recursively")
    parser.add_argument("--arrival-ms", type=float, required=True, help="Interval between arrivals")
    parser.add_argument("--queue-capacity", type=int, required=True, help="Waiting slots excluding the worker")
    parser.add_argument("--deadline-ms", type=float, required=True, help="Maximum arrival-to-completion latency")
    parser.add_argument("--cycles", type=int, default=1, help="Repeat the same sorted image sequence")
    parser.add_argument("--out", type=Path, required=True, help="Directory for trace.csv and summary.json")
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--service-ms", type=float, help="Assumed constant worker time; no model execution")
    timing.add_argument("--timings", type=Path, help="Measured per-frame timings CSV (file,service_ms)")
    args = parser.parse_args()
    try:
        names = image_names(args.images)
        if args.timings is None:
            durations = dict.fromkeys(names, positive_ms(args.service_ms))
            mode = "fixed_service"
        else:
            durations = load_timings(args.timings, names)
            mode = "measured_timing_csv"
        trace = simulate(names, durations, args.arrival_ms, args.queue_capacity, args.deadline_ms, args.cycles)
        summary = {
            "service_mode": mode,
            "unique_images": len(names),
            "cycles": args.cycles,
            "arrival_interval_ms": args.arrival_ms,
            "queue_capacity_waiting": args.queue_capacity,
            "deadline_ms": args.deadline_ms,
            **summarize(trace),
        }
        if mode == "fixed_service":
            summary["assumed_service_ms"] = args.service_ms
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "trace.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            writer.writerows(trace)
        report = json.dumps(summary, indent=2, ensure_ascii=False)
        (args.out / "summary.json").write_text(report + "\n", encoding="utf-8")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Replay failed: {error}\n")
    print(report)


if __name__ == "__main__":
    main()
