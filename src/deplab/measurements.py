from __future__ import annotations

import os
import shutil
from pathlib import Path

from .models import CacheSnapshot, MachineSnapshot, ResourceMetrics, StageResult, utc_now


def capture_machine(path: Path) -> MachineSnapshot:
    path.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(path)
    memory = _linux_memory()
    network = _linux_network()
    try:
        load_1m = os.getloadavg()[0]
    except (AttributeError, OSError):
        load_1m = None
    return MachineSnapshot(
        captured_at=utc_now(),
        cpu_count=os.cpu_count(),
        load_1m=load_1m,
        memory_total_bytes=memory.get("MemTotal"),
        memory_available_bytes=memory.get("MemAvailable"),
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        network_received_bytes=network[0] if network else None,
        network_transmitted_bytes=network[1] if network else None,
    )


def capture_cache(path: Path, measure_contents: bool = True) -> CacheSnapshot:
    if not path.exists():
        return CacheSnapshot(str(path), False, 0, 0)
    if not measure_contents:
        return CacheSnapshot(str(path), True, None, None)
    count = 0
    size = 0
    for root, _, files in os.walk(path, onerror=lambda _: None):
        for filename in files:
            try:
                size += (Path(root) / filename).stat().st_size
                count += 1
            except OSError:
                continue
    return CacheSnapshot(str(path), True, count, size)


def start_resource_metrics(
    run_root: Path,
    cache_dir: Path,
    measure_cache_contents: bool = True,
) -> ResourceMetrics:
    cache = capture_cache(cache_dir, measure_cache_contents)
    state = "populated" if _cache_has_entries(cache_dir) else "empty"
    return ResourceMetrics(
        measurement_scope=(
            "Process-tree peak RSS is per stage. Disk, memory, load and network counters "
            "are host-level snapshots and can include unrelated activity. "
            + (
                "Cache file count and size are exact."
                if measure_cache_contents
                else "Cache file count and size are omitted because a shared cache is used concurrently."
            )
        ),
        machine_before=capture_machine(run_root),
        cache_before=cache,
        cache_state_before=state,
    )


def finish_resource_metrics(
    metrics: ResourceMetrics,
    run_root: Path,
    cache_dir: Path,
    stages: list[StageResult],
    measure_cache_contents: bool = True,
) -> None:
    metrics.machine_after = capture_machine(run_root)
    metrics.cache_after = capture_cache(cache_dir, measure_cache_contents)
    if (
        metrics.cache_after.size_bytes is not None
        and metrics.cache_before.size_bytes is not None
    ):
        metrics.cache_size_change_bytes = (
            metrics.cache_after.size_bytes - metrics.cache_before.size_bytes
        )
    else:
        metrics.cache_size_change_bytes = None
    metrics.disk_free_change_bytes = (
        metrics.machine_after.disk_free_bytes - metrics.machine_before.disk_free_bytes
    )
    metrics.host_network_received_change_bytes = _optional_delta(
        metrics.machine_after.network_received_bytes,
        metrics.machine_before.network_received_bytes,
    )
    metrics.host_network_transmitted_change_bytes = _optional_delta(
        metrics.machine_after.network_transmitted_bytes,
        metrics.machine_before.network_transmitted_bytes,
    )
    peaks = [stage.peak_rss_bytes for stage in stages if stage.peak_rss_bytes is not None]
    metrics.peak_stage_rss_bytes = max(peaks) if peaks else None


def process_tree_rss_bytes(root_pid: int) -> int | None:
    """Return current RSS for a Linux process and all descendants."""
    proc = Path("/proc")
    if not proc.exists():
        return None
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            after_name = stat[stat.rfind(")") + 2 :].split()
            parents[pid] = int(after_name[1])
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss[pid] = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    included = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in included and pid not in included:
                included.add(pid)
                changed = True
    values = [rss[pid] for pid in included if pid in rss]
    return sum(values) if values else None


def _linux_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def _linux_network() -> tuple[int, int] | None:
    received = 0
    transmitted = 0
    found = False
    try:
        lines = Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]
        for line in lines:
            interface, counters = line.split(":", 1)
            if interface.strip() == "lo":
                continue
            values = counters.split()
            received += int(values[0])
            transmitted += int(values[8])
            found = True
    except (OSError, ValueError, IndexError):
        return None
    return (received, transmitted) if found else None


def _optional_delta(after: int | None, before: int | None) -> int | None:
    return after - before if after is not None and before is not None else None


def _cache_has_entries(path: Path) -> bool:
    try:
        return path.exists() and next(path.iterdir(), None) is not None
    except OSError:
        return False
