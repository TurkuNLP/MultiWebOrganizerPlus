from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

from label_pipeline_lib.common import LOGGER


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def detect_num_gpus(preferred: int | None = None) -> int:
    """Return the number of GPUs whose compute should be accounted for.

    When ``preferred`` is supplied, it represents the number of GPUs configured
    for this pipeline run (currently ``tensor_parallel_size``) and is therefore
    the correct value for GPU-hour accounting even if the job can see additional
    devices.

    If no configured count is supplied, fall back to visible-device detection.
    """
    if preferred is not None:
        if preferred < 1:
            raise ValueError("preferred GPU count must be >= 1")
        return preferred

    try:
        import torch

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            if count > 0:
                return count
    except (ImportError, RuntimeError):
        pass

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if devices:
            return len(devices)

    return 1


def start_compute_logger(
    *,
    interval: float,
    num_gpus: int,
    get_progress: Callable[[], tuple[int, int | None]],
    stop_event: threading.Event | None = None,
) -> threading.Event:
    """Start a daemon thread that periodically logs elapsed compute and ETA.

    ``get_progress`` must return ``(processed_this_run, total_to_process_this_run)``.
    The total may be ``None`` for streamed inputs whose size is unknown.
    """
    if interval <= 0:
        raise ValueError("compute logging interval must be > 0")
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")

    if stop_event is None:
        stop_event = threading.Event()

    start_time = time.monotonic()

    def worker() -> None:
        LOGGER.info(
            "Compute logger started: logging every %.1f seconds; tracking %d GPU(s)",
            interval,
            num_gpus,
        )
        while not stop_event.wait(interval):
            elapsed = time.monotonic() - start_time
            try:
                processed, total = get_progress()
            except Exception as exc:  # Logging must not terminate model inference.
                LOGGER.exception("Compute logger: failed to fetch progress: %s", exc)
                continue

            if processed < 0 or (total is not None and total < 0):
                LOGGER.error(
                    "Compute logger received invalid progress: processed=%d total=%s",
                    processed,
                    total,
                )
                continue

            gpu_hours_used = (elapsed * float(num_gpus)) / 3600.0
            summary = [
                f"elapsed={format_duration(elapsed)}",
                f"gpus={num_gpus}",
                f"gpu-hours-used={gpu_hours_used:.4f}",
                f"processed={processed}/{total if total is not None else 'unknown'}",
            ]

            if total is not None and total > 0 and processed > 0:
                rate = processed / elapsed if elapsed > 0 else 0.0
                if rate > 0:
                    remaining = max(0, total - processed)
                    seconds_left = remaining / rate
                    gpu_hours_left = (seconds_left * float(num_gpus)) / 3600.0
                    summary.append(f"eta={format_duration(seconds_left)}")
                    summary.append(f"gpu-hours-remaining={gpu_hours_left:.4f}")
                else:
                    summary.append("eta=unknown (rate=0)")

            LOGGER.info("Compute: %s", ", ".join(summary))

        LOGGER.info("Compute logger stopped")

    thread = threading.Thread(target=worker, name="compute-logger", daemon=True)
    thread.start()
    return stop_event
