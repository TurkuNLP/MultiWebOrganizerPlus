from __future__ import annotations

import threading
import time
from typing import Callable, Optional

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


def detect_num_gpus(preferred: Optional[int] = None) -> int:
    """Detect available GPU devices.

    Strategy (in order):
    - Try to use PyTorch if available (torch.cuda.device_count()).
    - Fall back to CUDA_VISIBLE_DEVICES env var (comma-separated indices/UUIDs).
    - Finally return preferred if provided, otherwise 1.
    """
    try:
        import torch

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            try:
                cnt = int(torch.cuda.device_count())
                if cnt > 0:
                    return cnt
            except Exception:
                # Fall through to other methods
                pass
    except Exception:
        # PyTorch not available or failed; continue
        pass

    # Check environment variable
    try:
        import os

        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env:
            # remove empty entries
            parts = [p for p in env.split(",") if p.strip() != ""]
            if parts:
                try:
                    return max(1, len(parts))
                except Exception:
                    pass
    except Exception:
        pass

    # Fallbacks
    if preferred is not None and preferred > 0:
        return preferred
    return 1


def start_compute_logger(
    *,
    interval: float,
    num_gpus: int,
    get_progress: Callable[[], tuple[int, int]],
    stop_event: Optional[threading.Event] = None,
) -> threading.Event:
    """Start a background thread that periodically logs compute usage.

    - interval: seconds between log messages
    - num_gpus: number of GPUs assumed in use
    - get_progress: callable returning (processed, total)

    Returns the threading.Event used to stop the logger. The caller may set
    this event to stop the background thread; it will also be set when the
    thread finishes.
    """

    if stop_event is None:
        stop_event = threading.Event()

    start_time = time.time()

    def worker() -> None:
        LOGGER.info(
            "Compute logger started: logging every %.1f seconds; tracking %d GPU(s)",
            interval,
            num_gpus,
        )
        while not stop_event.wait(interval):
            now = time.time()
            elapsed = now - start_time
            try:
                processed, total = get_progress()
            except Exception as exc:  # defensive
                LOGGER.exception("Compute logger: failed to fetch progress: %s", exc)
                continue

            # Basic GPU-hours consumed so far
            gpu_hours_used = (elapsed * float(num_gpus)) / 3600.0

            summary = [
                f"elapsed={format_duration(elapsed)}",
                f"gpus={num_gpus}",
                f"gpu-hours-used={gpu_hours_used:.4f}",
            ]

            remaining_msg = ""
            if total and processed > 0:
                rate = processed / elapsed if elapsed > 0 else 0.0
                if rate > 0:
                    remaining = total - processed
                    secs_left = remaining / rate
                    gpu_hours_left = (secs_left * float(num_gpus)) / 3600.0
                    summary.append(f"processed={processed}/{total}")
                    summary.append(f"eta={format_duration(secs_left)}")
                    summary.append(f"gpu-hours-remaining={gpu_hours_left:.4f}")
                else:
                    summary.append(f"processed={processed}/{total}")
                    summary.append("eta=unknown (rate=0)")
            else:
                # When total is unknown or nothing processed yet, just report processed if available
                if total:
                    summary.append(f"processed={processed}/{total}")
                else:
                    summary.append(f"processed={processed}")

            LOGGER.info("Compute: %s", ", ".join(summary))

        # Mark stop_event when exiting
        stop_event.set()
        LOGGER.info("Compute logger stopped")

    thread = threading.Thread(target=worker, name="compute-logger", daemon=True)
    thread.start()
    return stop_event
