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

import json
import os
import socket
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}
_state_lock = threading.Lock()
_write_lock = threading.Lock()
_configured = False
_enabled = False
_output_dir: Optional[Path] = None
_min_duration_us = 0
_step: Optional[int] = None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def configure(
    *,
    enabled: Optional[bool] = None,
    output_dir: Optional[str] = None,
    min_duration_ms: Optional[float] = None,
) -> None:
    """Configure lightweight JSONL timeline tracing for the current process."""
    global _configured, _enabled, _output_dir, _min_duration_us

    with _state_lock:
        env_enabled = os.environ.get("RLINF_TIMELINE")
        env_output_dir = os.environ.get("RLINF_TIMELINE_DIR")
        env_min_ms = os.environ.get("RLINF_TIMELINE_MIN_MS")

        if enabled is None:
            enabled = _as_bool(env_enabled, False)
        elif env_enabled is not None:
            enabled = _as_bool(env_enabled, enabled)

        if output_dir is None:
            output_dir = env_output_dir
        if min_duration_ms is None and env_min_ms is not None:
            min_duration_ms = float(env_min_ms)

        _enabled = bool(enabled)
        _output_dir = Path(output_dir).expanduser() if output_dir else None
        _min_duration_us = int(float(min_duration_ms or 0.0) * 1000)
        if _enabled and _output_dir is not None:
            _output_dir.mkdir(parents=True, exist_ok=True)
        _configured = True


def configure_from_cfg(cfg: Any) -> None:
    """Configure tracing from a Hydra config, with env vars as overrides."""
    if cfg is None:
        configure()
        return

    runner = getattr(cfg, "runner", None)
    timeline = None
    if runner is not None and hasattr(runner, "get"):
        timeline = runner.get("timeline", None)

    enabled = None
    output_dir = None
    min_duration_ms = None
    if timeline is not None and hasattr(timeline, "get"):
        enabled = timeline.get("enabled", None)
        output_dir = timeline.get("output_dir", None)
        min_duration_ms = timeline.get("min_duration_ms", None)

    if output_dir is None and runner is not None:
        logger = getattr(runner, "logger", None)
        log_path = getattr(logger, "log_path", None)
        if log_path:
            output_dir = os.path.join(str(log_path), "timeline")

    configure(
        enabled=enabled,
        output_dir=output_dir,
        min_duration_ms=min_duration_ms,
    )


def set_step(step: Optional[int]) -> None:
    global _step
    _step = int(step) if step is not None else None


def is_enabled() -> bool:
    if not _configured:
        configure()
    return _enabled and _output_dir is not None


def _trace_path() -> Path:
    assert _output_dir is not None
    worker = os.environ.get("WORKER_NAME", "driver").replace("/", "_")
    rank = os.environ.get("RANK", "na")
    return _output_dir / f"trace_{socket.gethostname()}_{worker}_rank{rank}_pid{os.getpid()}.jsonl"


def _default_args(extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    args = {
        "step": _step,
        "hostname": socket.gethostname(),
        "worker_name": os.environ.get("WORKER_NAME"),
        "rank": os.environ.get("RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "cluster_node_rank": os.environ.get("CLUSTER_NODE_RANK"),
        "local_accelerator_rank": os.environ.get("LOCAL_ACCELERATOR_RANK"),
        "accelerator_type": os.environ.get("ACCELERATOR_TYPE"),
        "accelerator_model": os.environ.get("ACCELERATOR_MODEL"),
    }
    if extra:
        args.update(extra)
    return {k: v for k, v in args.items() if v is not None}


def write_event(
    name: str,
    *,
    category: str,
    start_time_ns: int,
    duration_ns: int,
    args: Optional[dict[str, Any]] = None,
) -> None:
    if not is_enabled():
        return
    duration_us = duration_ns // 1000
    if duration_us < _min_duration_us:
        return

    event = {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": start_time_ns // 1000,
        "dur": duration_us,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "args": _default_args(args),
    }
    line = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
    with _write_lock:
        with _trace_path().open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


@contextmanager
def span(
    name: str,
    *,
    category: str = "default",
    args: Optional[dict[str, Any]] = None,
) -> Iterator[None]:
    if not is_enabled():
        yield
        return

    start_wall_ns = time.time_ns()
    start_perf_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        write_event(
            name,
            category=category,
            start_time_ns=start_wall_ns,
            duration_ns=time.perf_counter_ns() - start_perf_ns,
            args=args,
        )


def worker_span(worker: Any, name: str, *, category: str = "worker"):
    cfg = getattr(worker, "cfg", None)
    if cfg is not None:
        configure_from_cfg(cfg)
    elif not _configured:
        configure()

    if not is_enabled():
        return nullcontext()

    worker_step = getattr(worker, "_timeline_step", None)
    if worker_step is None:
        worker_step = getattr(worker, "version", None)
    set_step(worker_step)

    args = {
        "group_name": getattr(worker, "_group_name", None),
        "worker_rank": getattr(worker, "_rank", None),
        "worker_world_size": getattr(worker, "_world_size", None),
        "worker_class": worker.__class__.__name__,
    }
    return span(name, category=category, args=args)
