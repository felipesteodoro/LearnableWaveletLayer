"""
GPU Queue Monitor — Rich terminal dashboard.

Run in a separate terminal while the queue is executing:

    python -m tests.financial.queue.dashboard
    # or
    python tests/financial/queue/dashboard.py [--status path/to/queue_status.json]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_STATUS = (
    Path(__file__).parent.parent / "results" / "queue_status.json"
)
REFRESH_RATE = 2.0  # seconds


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fmt_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    return str(timedelta(seconds=int(seconds)))[2:]  # strip leading "0:"


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_header(data: dict) -> Panel:
    s = data["summary"]
    total = s["total"] or 1
    done = s["done"]
    failed = s["failed"]
    running = s["running"]
    retrying = s["retrying"]
    pending = s["pending"]

    pct = 100 * done / total
    bar_len = 28
    filled = int(pct / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    queue_elapsed = time.time() - data["queue_start_time"]
    eta_str = "—"
    if done > 0 and pending + running + retrying > 0:
        rate = done / queue_elapsed          # jobs/sec
        remaining = pending + running + retrying
        eta_sec = remaining / rate
        eta_str = _fmt_elapsed(eta_sec)

    row = Text(justify="center")
    row.append(f" [{bar}] {pct:.1f}%   ", style="bold cyan")
    row.append(f"✓ {done} done  ", style="bold green")
    row.append(f"▶ {running} running  ", style="bold yellow")
    row.append(f"↻ {retrying} retrying  ", style="bold orange3")
    row.append(f"✗ {failed} failed  ", style="bold red")
    row.append(f"◌ {pending} pending  ", style="dim")
    row.append(f" | elapsed {_fmt_elapsed(queue_elapsed)}  eta {eta_str}", style="dim")

    return Panel(row, title="[bold blue]GPU Job Queue Monitor[/bold blue]", border_style="blue")


def _render_gpu_table(data: dict) -> Panel:
    running_by_gpu: dict[int, dict] = {
        j["gpu_id"]: j for j in data["jobs"] if j["status"] == "running"
    }
    retrying_by_gpu: dict[int, dict] = {
        j["gpu_id"]: j for j in data["jobs"] if j["status"] == "retrying"
    }

    # Infer total GPU count from jobs or default to 7
    all_gpu_ids = sorted(
        {j["gpu_id"] for j in data["jobs"] if j["gpu_id"] is not None}
    )
    n_gpus = max(max(all_gpu_ids, default=6) + 1, 7)

    table = Table(box=box.ROUNDED, border_style="cyan", expand=True)
    table.add_column("GPU", justify="center", width=5, style="bold")
    table.add_column("Status", width=12)
    table.add_column("Ticker", width=9)
    table.add_column("Model", width=12)
    table.add_column("Mode", width=16)
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
                j["ticker"],
                j["model_name"],
                j["mode"],
                _fmt_elapsed(j["elapsed"]),
                retry_str,
                str(j["pid"] or "—"),
            )
        elif gid in retrying_by_gpu:
            j = retrying_by_gpu[gid]
            table.add_row(
                f"[orange3]{gid}[/orange3]",
                "[bold orange3]↻ RETRY[/]",
                j["ticker"],
                j["model_name"],
                j["mode"],
                _fmt_elapsed(j["elapsed"]),
                f"{j['retry_count']}/{j['max_retries']}",
                "—",
            )
        else:
            table.add_row(
                str(gid), "[dim]○ idle[/dim]", "—", "—", "—", "—", "—", "—"
            )

    return Panel(table, title="[bold]GPU Status[/bold]", border_style="cyan")


def _render_recent(data: dict, n: int = 10) -> Panel:
    finished = sorted(
        [j for j in data["jobs"] if j["status"] in ("done", "failed")],
        key=lambda j: j.get("end_time") or 0,
        reverse=True,
    )[:n]

    table = Table(box=box.SIMPLE, expand=True, show_header=True)
    table.add_column("", width=2)
    table.add_column("Job", width=32)
    table.add_column("GPU", width=4, justify="center")
    table.add_column("Duration", width=9, justify="right")
    table.add_column("Ended", width=9, justify="right", style="dim")

    for j in finished:
        if j["status"] == "done":
            icon, style = "✓", "green"
        else:
            icon, style = "✗", "red"

        start = j.get("start_time") or 0
        end = j.get("end_time") or 0
        duration = end - start if start and end else None

        table.add_row(
            f"[{style}]{icon}[/]",
            f"[{style}]{j['name']}[/]",
            str(j["gpu_id"] if j["gpu_id"] is not None else "—"),
            _fmt_elapsed(duration),
            _fmt_ts(j.get("end_time")),
        )

    return Panel(table, title="[bold]Recent Completions[/bold]", border_style="green")


def _render_errors(data: dict) -> Panel:
    failed = [j for j in data["jobs"] if j["status"] == "failed" and j.get("error_msg")]

    if not failed:
        return Panel(
            "[dim]No failures.[/dim]",
            title=f"[bold]Errors (0)[/bold]",
            border_style="dim",
        )

    lines: list[str] = []
    for j in failed[-4:]:
        lines.append(f"[bold red]{j['name']}[/bold red]  [dim]retry {j['retry_count']}/{j['max_retries']}[/dim]")
        excerpt = (j.get("error_msg") or "").strip().splitlines()
        # Show last 5 lines of error
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


def _build_layout(data: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(_render_header(data), name="header", size=3),
        Layout(name="body"),
        Layout(_render_errors(data), name="errors", size=14),
    )
    layout["body"].split_row(
        Layout(_render_gpu_table(data), name="gpus", ratio=3),
        Layout(_render_recent(data), name="recent", ratio=2),
    )
    return layout


def _waiting_panel() -> Panel:
    return Panel(
        "[dim]Waiting for queue_status.json to appear…[/dim]\n"
        "[dim]Start the job queue with:  python run_dl_queue.py[/dim]",
        title="[bold blue]GPU Job Queue Monitor[/bold blue]",
        border_style="blue",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_dashboard(status_file: Path = DEFAULT_STATUS, refresh: float = REFRESH_RATE):
    console = Console()

    def render():
        data = _load(status_file)
        if data is None:
            return _waiting_panel()
        return _build_layout(data)

    with Live(render(), refresh_per_second=1 / refresh, screen=True, console=console) as live:
        try:
            while True:
                live.update(render())
                time.sleep(refresh)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="GPU Job Queue Monitor")
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS,
        help="Path to queue_status.json",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=REFRESH_RATE,
        help="Refresh interval in seconds (default: 2)",
    )
    args = parser.parse_args()
    run_dashboard(args.status, args.refresh)


if __name__ == "__main__":
    main()
