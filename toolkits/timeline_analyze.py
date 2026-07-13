# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def load_events(timeline_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(timeline_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad JSON in {path}:{line_no}: {exc}") from exc
    events.sort(key=lambda event: (event.get("ts", 0), event.get("pid", 0)))
    return events


def event_step(event: dict[str, Any]) -> str:
    step = event.get("args", {}).get("step")
    return "" if step is None else str(step)


def event_worker(event: dict[str, Any]) -> str:
    args = event.get("args", {})
    return (
        args.get("worker_class")
        or args.get("worker_name")
        or event.get("cat")
        or "unknown"
    )


def summarize(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for event in events:
        grouped[(event_step(event), event_worker(event), event["name"])].append(
            event.get("dur", 0) / 1_000_000.0
        )

    rows = []
    for (step, worker, name), durations in grouped.items():
        rows.append(
            {
                "step": step,
                "worker": worker,
                "name": name,
                "count": len(durations),
                "sum_s": sum(durations),
                "max_s": max(durations),
                "p50_s": median(durations),
                "p95_s": percentile(durations, 0.95),
            }
        )
    rows.sort(key=lambda row: (row["step"], -row["max_s"], row["worker"], row["name"]))
    return rows


def summarize_step_wall(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bounds: dict[str, list[float]] = defaultdict(lambda: [float("inf"), 0.0])
    for event in events:
        step = event_step(event)
        if step == "":
            continue
        start_s = event.get("ts", 0) / 1_000_000.0
        end_s = start_s + event.get("dur", 0) / 1_000_000.0
        bounds[step][0] = min(bounds[step][0], start_s)
        bounds[step][1] = max(bounds[step][1], end_s)
    rows = []
    for step, (start_s, end_s) in bounds.items():
        rows.append({"step": step, "timeline_wall_s": end_s - start_s})
    rows.sort(key=lambda row: int(row["step"]) if row["step"].isdigit() else row["step"])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge RLinf timeline JSONL files and print timing summaries."
    )
    parser.add_argument("timeline_dir", type=Path)
    parser.add_argument("--chrome-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    events = load_events(args.timeline_dir)
    if not events:
        raise SystemExit(f"No timeline events found in {args.timeline_dir}")

    chrome_out = args.chrome_out or args.timeline_dir / "chrome_trace.json"
    chrome_out.write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}),
        encoding="utf-8",
    )

    rows = summarize(events)
    if args.summary_out is not None:
        write_csv(args.summary_out, rows)

    print(f"events={len(events)}")
    print(f"chrome_trace={chrome_out}")
    for row in summarize_step_wall(events)[: args.top]:
        print(f"step {row['step']}: timeline_wall_s={row['timeline_wall_s']:.3f}")

    print("\nTop spans by max_s:")
    for row in sorted(rows, key=lambda item: item["max_s"], reverse=True)[: args.top]:
        print(
            f"step={row['step'] or '-'} worker={row['worker']} "
            f"name={row['name']} count={row['count']} "
            f"sum_s={row['sum_s']:.3f} max_s={row['max_s']:.3f} "
            f"p50_s={row['p50_s']:.3f} p95_s={row['p95_s']:.3f}"
        )


if __name__ == "__main__":
    main()
