"""
GPU Queue Monitor — Rich terminal dashboard (Synthetic experiment).

Run in a separate terminal while the queue is executing:

    python tests/synthetic/gpu_queue/dashboard.py
    python tests/synthetic/gpu_queue/dashboard.py --status results/2026-05-09/queue_status.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

REFRESH_RATE = 2.0  # seconds

_RESULTS_BASE = Path(__file__).parent.parent / "results"
_SYNTHETIC_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _find_latest_status() -> Path:
    dated = sorted(_RESULTS_BASE.glob("????-??-??*"), reverse=True)
    for folder in dated:
        candidate = folder / "queue_status.json"
        if candidate.exists():
            return candidate
    return _RESULTS_BASE / "queue_status.json"


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fmt_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    return str(timedelta(seconds=int(seconds)))[2:]


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _load_metrics(results_dir: Path, job: dict) -> dict:
    model_name = job.get("model_name", "")
    mode       = job.get("mode", "")
    cfg_idx    = job.get("config_idx", 0)
    path = results_dir / f"{model_name}_{mode}" / f"cfg{cfg_idx:03d}" / "metrics.json"
    try:
        with open(path) as f:
            m = json.load(f)
        return {
            "test_rmse": m.get("test_rmse"),
            "test_r2":   m.get("test_r2"),
            "test_mae":  m.get("test_mae"),
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {"test_rmse": None, "test_r2": None, "test_mae": None}


def _fmt_rmse(v: Optional[float]) -> Text:
    if v is None:
        return Text("—", style="dim")
    color = "green" if v < 0.1 else ("yellow" if v < 0.3 else "red")
    return Text(f"{v:.5f}", style=color)


def _fmt_r2(v: Optional[float]) -> Text:
    if v is None:
        return Text("—", style="dim")
    color = "green" if v > 0.7 else ("yellow" if v > 0.3 else "red")
    return Text(f"{v:.4f}", style=color)


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _render_header(data: dict) -> Panel:
    s = data["summary"]
    total = s["total"] or 1
    done, failed = s["done"], s["failed"]
    running, retrying, pending = s["running"], s["retrying"], s["pending"]

    pct = 100 * done / total
    bar_len = 28
    filled = int(pct / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    queue_elapsed = time.time() - data["queue_start_time"]
    eta_str = "—"
    if done > 0:
        rate = done / queue_elapsed
        unstarted = pending + retrying
        if unstarted > 0:
            eta_str = _fmt_elapsed(unstarted / rate)
        elif running > 0:
            eta_str = "finishing…"

    row = Text(justify="center")
    row.append(f" [{bar}] {pct:.1f}%   ", style="bold cyan")
    row.append(f"✓ {done} done  ", style="bold green")
    row.append(f"▶ {running} running  ", style="bold yellow")
    row.append(f"↻ {retrying} retrying  ", style="bold orange3")
    row.append(f"✗ {failed} failed  ", style="bold red")
    row.append(f"◌ {pending} pending  ", style="dim")
    row.append(f" | elapsed {_fmt_elapsed(queue_elapsed)}  eta {eta_str}", style="dim")

    return Panel(row, title="[bold blue]Synthetic GPU Job Queue Monitor[/bold blue]", border_style="blue")


def _render_config_summary() -> Panel:
    """Show the synthetic signal and training configuration at a glance."""
    try:
        import sys
        _cfg_dir = str(_SYNTHETIC_DIR / "config")
        if _cfg_dir not in sys.path:
            sys.path.insert(0, _cfg_dir)
        from experiment_config import SYNTHETIC_SIGNAL_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG
    except Exception:
        return Panel("[dim]Config unavailable[/dim]", title="[bold]Experiment Config[/bold]", border_style="magenta")

    sig = SYNTHETIC_SIGNAL_CONFIG
    train = DL_TRAINING_CONFIG
    wav = LEARNED_WAVELET_CONFIG

    row1 = Text()
    row1.append("Signal  ", style="bold")
    row1.append(f"n_samples={sig.get('n_samples', '?')}  "
                f"seq_len={sig.get('sequence_length', '?')}  "
                f"noise={sig.get('noise_level', '?')}  "
                f"harmonics={sig.get('n_harmonics', '?')}  "
                f"regimes={sig.get('regime_changes', '?')}")

    row2 = Text()
    row2.append("Training  ", style="bold")
    row2.append(f"epochs={train.get('epochs', '?')}  "
                f"batch={train.get('batch_size', '?')}  "
                f"lr_patience={train.get('reduce_lr_patience', '?')}  "
                f"es_patience={train.get('early_stopping_patience', '?')}")

    row3 = Text()
    row3.append("Wavelet  ", style="bold")
    row3.append(f"levels={wav.get('levels', '?')}  "
                f"kernel_size={wav.get('kernel_size', '?')}  "
                f"net_units={wav.get('wavelet_net_units', '?')}  "
                f"align={wav.get('align', '?')}")

    return Panel(Group(row1, row2, row3), title="[bold]Experiment Config[/bold]", border_style="magenta")


def _render_gpu_table(data: dict) -> Panel:
    running_by_gpu: dict[int, dict] = {
        j["gpu_id"]: j for j in data["jobs"] if j["status"] == "running"
    }
    retrying_by_gpu: dict[int, dict] = {
        j["gpu_id"]: j for j in data["jobs"] if j["status"] == "retrying"
    }

    all_gpu_ids = sorted({j["gpu_id"] for j in data["jobs"] if j["gpu_id"] is not None})
    n_gpus = max(max(all_gpu_ids, default=6) + 1, 7)

    table = Table(box=box.ROUNDED, border_style="cyan", expand=True)
    table.add_column("GPU", justify="center", width=5, style="bold")
    table.add_column("Status", width=12)
    table.add_column("Model", width=12)
    table.add_column("Mode", width=26)
    table.add_column("Cfg", width=5, justify="center")
    table.add_column("Elapsed", width=9, justify="right")
    table.add_column("Retry", width=7, justify="center")
    table.add_column("PID", width=7, justify="right", style="dim")

    for gid in range(n_gpus):
        if gid in running_by_gpu:
            j = running_by_gpu[gid]
            retry_str = f"{j['retry_count']}/{j['max_retries']}" if j["retry_count"] else "—"
            table.add_row(
                f"[yellow]{gid}[/yellow]",
                "[bold green]● RUNNING[/]",
                j["model_name"],
                j["mode"],
                str(j["config_idx"]),
                _fmt_elapsed(j["elapsed"]),
                retry_str,
                str(j["pid"] or "—"),
            )
        elif gid in retrying_by_gpu:
            j = retrying_by_gpu[gid]
            table.add_row(
                f"[orange3]{gid}[/orange3]",
                "[bold orange3]↻ RETRY[/]",
                j["model_name"],
                j["mode"],
                str(j["config_idx"]),
                _fmt_elapsed(j["elapsed"]),
                f"{j['retry_count']}/{j['max_retries']}",
                "—",
            )
        else:
            table.add_row(str(gid), "[dim]○ idle[/dim]", "—", "—", "—", "—", "—", "—")

    return Panel(table, title="[bold]GPU Status[/bold]", border_style="cyan")


def _render_recent(data: dict, results_dir: Path, n: int = 12) -> Panel:
    finished = sorted(
        [j for j in data["jobs"] if j["status"] in ("done", "failed")],
        key=lambda j: j.get("end_time") or 0,
        reverse=True,
    )[:n]

    table = Table(box=box.SIMPLE, expand=True, show_header=True)
    table.add_column("", width=2)
    table.add_column("Job", width=40)
    table.add_column("GPU", width=4, justify="center")
    table.add_column("Duration", width=9, justify="right")
    table.add_column("Test RMSE", width=11, justify="right")
    table.add_column("Test R²", width=9, justify="right")
    table.add_column("Ended", width=9, justify="right", style="dim")

    for j in finished:
        if j["status"] == "done":
            icon, style = "✓", "green"
            m = _load_metrics(results_dir, j)
        else:
            icon, style = "✗", "red"
            m = {"test_rmse": None, "test_r2": None}

        start = j.get("start_time") or 0
        end   = j.get("end_time") or 0
        duration = end - start if start and end else None

        table.add_row(
            f"[{style}]{icon}[/]",
            f"[{style}]{j['name']}[/]",
            str(j["gpu_id"] if j["gpu_id"] is not None else "—"),
            _fmt_elapsed(duration),
            _fmt_rmse(m["test_rmse"]),
            _fmt_r2(m["test_r2"]),
            _fmt_ts(j.get("end_time")),
        )

    return Panel(table, title="[bold]Recent Completions[/bold]", border_style="green")


def _render_errors(data: dict) -> Panel:
    failed = [j for j in data["jobs"] if j["status"] == "failed" and j.get("error_msg")]

    if not failed:
        return Panel("[dim]No failures.[/dim]", title="[bold]Errors (0)[/bold]", border_style="dim")

    lines: list[str] = []
    for j in failed[-2:]:
        lines.append(f"[bold red]{j['name']}[/bold red]  [dim]retry {j['retry_count']}/{j['max_retries']}[/dim]")
        excerpt = (j.get("error_msg") or "").strip().splitlines()
        for line in excerpt[-5:]:
            lines.append(f"  [dim]{line}[/dim]")
        lines.append("")

    return Panel(
        "\n".join(lines).strip(),
        title=f"[bold red]Errors ({len(failed)})[/bold red]",
        border_style="red",
    )


# ---------------------------------------------------------------------------
# Layout assembly
# ---------------------------------------------------------------------------


def _build_layout(data: dict, results_dir: Path) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(_render_header(data), name="header", size=3),
        Layout(_render_config_summary(), name="config", size=7),
        Layout(name="body"),
        Layout(_render_errors(data), name="errors", size=6),
    )
    layout["body"].split_row(
        Layout(_render_gpu_table(data), name="gpus", ratio=3),
        Layout(_render_recent(data, results_dir), name="recent", ratio=2),
    )
    return layout


def _waiting_panel() -> Panel:
    return Panel(
        "[dim]Waiting for queue_status.json to appear…[/dim]\n"
        "[dim]Start the job queue with:  python run_dl_queue.py[/dim]",
        title="[bold blue]Synthetic GPU Job Queue Monitor[/bold blue]",
        border_style="blue",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_dashboard(status_file: Path | None = None, refresh: float = REFRESH_RATE):
    if status_file is None:
        status_file = _find_latest_status()
    results_dir = status_file.parent
    console = Console()

    def render():
        data = _load(status_file)
        if data is None:
            return _waiting_panel()
        return _build_layout(data, results_dir)

    with Live(render(), refresh_per_second=1 / refresh, screen=True, console=console) as live:
        try:
            while True:
                live.update(render())
                time.sleep(refresh)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Synthetic GPU Job Queue Monitor")
    parser.add_argument("--status", type=Path, default=None,
                        help="Path to queue_status.json")
    parser.add_argument("--refresh", type=float, default=REFRESH_RATE,
                        help="Refresh interval in seconds (default: 2)")
    args = parser.parse_args()
    status = args.status or _find_latest_status()
    print(f"Dashboard monitoring: {status}")
    run_dashboard(status, args.refresh)


if __name__ == "__main__":
    main()
