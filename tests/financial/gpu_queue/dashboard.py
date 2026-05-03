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

REFRESH_RATE = 2.0  # seconds

_RESULTS_BASE = Path(__file__).parent.parent / "results"


def _find_latest_status() -> Path:
    """
    Encontra o queue_status.json mais recente dentro de results/.

    Procura primeiro por pastas datadas (YYYY-MM-DD_HHMMSS), que são as runs
    criadas pelo novo run_dl_queue.py. Se não houver nenhuma, cai no caminho
    legado results/queue_status.json para compatibilidade com runs antigas.
    """
    dated = sorted(_RESULTS_BASE.glob("????-??-??"), reverse=True)
    for folder in dated:
        candidate = folder / "queue_status.json"
        if candidate.exists():
            return candidate
    # Fallback legado: results/queue_status.json
    return _RESULTS_BASE / "queue_status.json"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _is_valid(model: str, mode: str, feature_mode: str) -> bool:
    """Mirror of run_dl_queue._is_valid — True if combination is scientifically sound."""
    if feature_mode == "features" and mode != "raw":
        return False  # features are already frequency filters; wavelet on top = confounded
    if model == "MLP" and mode != "raw":
        return False  # MLP flattens temporal structure, wavelet frontend is meaningless
    return True


def _fmt_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    return str(timedelta(seconds=int(seconds)))[2:]  # strip leading "0:"


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _load_oos_metrics(results_dir: Path, job: dict) -> dict:
    """Lê metrics.json do job e retorna oos_sharpe, oos_accuracy e oos_bh_sharpe (ou None)."""
    fmode = job.get("feature_mode", "features")
    ticker = job.get("ticker", "")
    key = f"{job.get('model_name', '')}_{job.get('mode', '')}"
    path = results_dir / fmode / ticker / key / "metrics.json"
    try:
        with open(path) as f:
            m = json.load(f)
        fin = m.get("fin_metrics", {})
        return {
            "oos_sharpe":    fin.get("oos_sharpe"),
            "oos_accuracy":  fin.get("oos_accuracy"),
            "oos_bh_sharpe": fin.get("oos_bh_sharpe"),
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {"oos_sharpe": None, "oos_accuracy": None, "oos_bh_sharpe": None}


def _fmt_sharpe(v: Optional[float]) -> Text:
    if v is None:
        return Text("—", style="dim")
    color = "green" if v > 0 else "red"
    return Text(f"{v:+.2f}", style=color)


def _fmt_acc(v: Optional[float]) -> Text:
    if v is None:
        return Text("—", style="dim")
    color = "green" if v >= 0.45 else ("yellow" if v >= 0.38 else "red")
    return Text(f"{v*100:.1f}%", style=color)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_validity_stats(data: dict) -> Panel:
    jobs = data["jobs"]
    total = len(jobs)

    valid_jobs   = [j for j in jobs if _is_valid(j["model_name"], j["mode"], j["feature_mode"])]
    invalid_jobs = [j for j in jobs if not _is_valid(j["model_name"], j["mode"], j["feature_mode"])]

    n_valid   = len(valid_jobs)
    n_invalid = len(invalid_jobs)
    pct_valid   = 100 * n_valid   / total if total else 0
    pct_invalid = 100 * n_invalid / total if total else 0

    n_feat_wavelet_total = sum(
        1 for j in invalid_jobs if j["feature_mode"] == "features" and j["mode"] != "raw"
    )
    n_mlp_wavelet_total = sum(
        1 for j in invalid_jobs if j["model_name"] == "MLP" and j["mode"] != "raw"
    )
    n_both = sum(
        1 for j in invalid_jobs
        if j["feature_mode"] == "features" and j["model_name"] == "MLP" and j["mode"] != "raw"
    )

    v_done    = sum(1 for j in valid_jobs if j["status"] == "done")
    v_running = sum(1 for j in valid_jobs if j["status"] == "running")
    v_failed  = sum(1 for j in valid_jobs if j["status"] == "failed")
    v_pending = sum(1 for j in valid_jobs if j["status"] in ("pending", "retrying"))

    row1 = Text()
    row1.append(f"Total: {total}   ", style="bold")
    row1.append(f"✓ Valid: {n_valid} ({pct_valid:.0f}%)   ", style="bold green")
    row1.append(f"✗ Invalid: {n_invalid} ({pct_invalid:.0f}%)", style="bold red")

    row2 = Text()
    row2.append("  Invalid reasons: ", style="dim")
    row2.append(f"features+wavelet = {n_feat_wavelet_total}", style="red")
    row2.append("  (EMA/MACD/BB são filtros de freq. — confundidor)", style="dim")
    row2.append(f"   MLP+wavelet = {n_mlp_wavelet_total}", style="red")
    row2.append("  (MLP faz Flatten — estrutura temporal ignorada)", style="dim")
    if n_both:
        row2.append(f"   ambos = {n_both}", style="dim red")

    row3 = Text()
    row3.append("  Valid progress: ", style="dim")
    row3.append(f"done {v_done}  ", style="green")
    row3.append(f"running {v_running}  ", style="yellow")
    row3.append(f"failed {v_failed}  ", style="red")
    row3.append(f"pending {v_pending}", style="dim")

    from rich.console import Group
    return Panel(Group(row1, row2, row3), title="[bold]Experiment Validity[/bold]", border_style="magenta")


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
    if done > 0:
        rate = done / queue_elapsed          # jobs/sec (parallel throughput)
        unstarted = pending + retrying
        if unstarted > 0:
            # Only count jobs not yet started — running ones are already in flight
            eta_sec = unstarted / rate
            eta_str = _fmt_elapsed(eta_sec)
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


def _render_recent(data: dict, results_dir: Path, n: int = 12) -> Panel:
    finished = sorted(
        [j for j in data["jobs"] if j["status"] in ("done", "failed")],
        key=lambda j: j.get("end_time") or 0,
        reverse=True,
    )[:n]

    table = Table(box=box.SIMPLE, expand=True, show_header=True)
    table.add_column("", width=2)
    table.add_column("Job", width=36)
    table.add_column("GPU", width=4, justify="center")
    table.add_column("Duration", width=9, justify="right")
    table.add_column("OOS Sharpe", width=11, justify="right")
    table.add_column("BH Sharpe", width=10, justify="right")
    table.add_column("OOS Acc", width=9, justify="right")
    table.add_column("Ended", width=9, justify="right", style="dim")

    for j in finished:
        if j["status"] == "done":
            icon, style = "✓", "green"
            oos = _load_oos_metrics(results_dir, j)
        else:
            icon, style = "✗", "red"
            oos = {"oos_sharpe": None, "oos_accuracy": None}

        start = j.get("start_time") or 0
        end = j.get("end_time") or 0
        duration = end - start if start and end else None

        table.add_row(
            f"[{style}]{icon}[/]",
            f"[{style}]{j['name']}[/]",
            str(j["gpu_id"] if j["gpu_id"] is not None else "—"),
            _fmt_elapsed(duration),
            _fmt_sharpe(oos["oos_sharpe"]),
            _fmt_sharpe(oos["oos_bh_sharpe"]),
            _fmt_acc(oos["oos_accuracy"]),
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


def _build_layout(data: dict, results_dir: Path) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(_render_header(data), name="header", size=3),
        Layout(_render_validity_stats(data), name="validity", size=7),
        Layout(name="body"),
        Layout(_render_errors(data), name="errors", size=14),
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
        title="[bold blue]GPU Job Queue Monitor[/bold blue]",
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
    parser = argparse.ArgumentParser(description="GPU Job Queue Monitor")
    parser.add_argument(
        "--status",
        type=Path,
        default=None,
        help="Path to queue_status.json (padrão: run mais recente em results/)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=REFRESH_RATE,
        help="Refresh interval in seconds (default: 2)",
    )
    args = parser.parse_args()
    status = args.status or _find_latest_status()
    print(f"Dashboard monitorando: {status}")
    run_dashboard(status, args.refresh)


if __name__ == "__main__":
    main()
