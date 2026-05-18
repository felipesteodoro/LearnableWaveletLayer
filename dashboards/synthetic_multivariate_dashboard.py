"""
Streamlit dashboard — Synthetic-Multivariate Experiment

Run:
    streamlit run dashboards/synthetic_multivariate_dashboard.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SYNTHETIC_DIR = Path(__file__).parent.parent / "tests" / "synthetic-multivariate"
RESULTS_BASE  = SYNTHETIC_DIR / "results"

st.set_page_config(
    page_title="Synthetic-Multivariate Experiment",
    page_icon="🌐",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def load_queue_status(run_id: str) -> dict | None:
    path = RESULTS_BASE / run_id / "queue_status.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_all_metrics(run_id: str) -> pd.DataFrame:
    rows = []
    pattern = RESULTS_BASE / run_id
    for metrics_file in pattern.rglob("metrics.json"):
        try:
            m = json.loads(metrics_file.read_text())
            rows.append(m)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    return df


def list_runs() -> list[str]:
    if not RESULTS_BASE.exists():
        return []
    runs = sorted([d.name for d in RESULTS_BASE.iterdir()
                   if d.is_dir() and d.name[0].isdigit()
                   and (d / "queue_status.json").exists()], reverse=True)
    return runs


def load_exp_config() -> dict:
    try:
        import sys
        cfg_path = str(SYNTHETIC_DIR / "config")
        if cfg_path not in sys.path:
            sys.path.insert(0, cfg_path)
        from experiment_config import SYNTHETIC_SIGNAL_CONFIG, MULTIVARIATE_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG
        return {
            "signal":  SYNTHETIC_SIGNAL_CONFIG,
            "multivariate": MULTIVARIATE_CONFIG,
            "training": DL_TRAINING_CONFIG,
            "wavelet": LEARNED_WAVELET_CONFIG,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("� Synthetic-Multivariate Experiment")
runs = list_runs()

if not runs:
    st.warning("Nenhuma run encontrada em `tests/synthetic-multivariate/results/`. Execute `run_dl_queue.py` primeiro.")
    st.stop()

run_id = st.sidebar.selectbox("Run ID", runs, index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)
metric_col   = st.sidebar.selectbox("Métrica principal", ["test_rmse", "test_mae", "test_r2"], index=0)
ascending    = metric_col != "test_r2"  # lower is better for RMSE/MAE, higher for R²

if auto_refresh:
    st.sidebar.caption("Atualizando a cada 10s…")

# Config expander
cfg = load_exp_config()
if cfg:
    with st.sidebar.expander("Configuração do experimento", expanded=False):
        st.json(cfg)

# ---------------------------------------------------------------------------
# Header — Queue status
# ---------------------------------------------------------------------------

status = load_queue_status(run_id)

st.title("� Synthetic-Multivariate Experiment Dashboard")
st.caption(f"Run: **{run_id}**  |  Atualizado em {pd.Timestamp.now().strftime('%H:%M:%S')}")

if status:
    s = status["summary"]
    total = max(s["total"], 1)
    pct   = s["done"] / total

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total",    s["total"])
    c2.metric("✅ Done",   s["done"],    delta=None)
    c3.metric("▶ Running", s["running"])
    c4.metric("↻ Retry",   s["retrying"])
    c5.metric("❌ Failed",  s["failed"])

    st.progress(pct, text=f"{pct*100:.1f}% concluído")

    elapsed = time.time() - status.get("queue_start_time", time.time())
    eta_str = "—"
    if s["done"] > 0 and (s["pending"] + s["retrying"]) > 0:
        rate = s["done"] / elapsed
        eta_sec = (s["pending"] + s["retrying"]) / rate
        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
    st.caption(f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s   |   ETA: {eta_str}")
else:
    st.warning("queue_status.json não encontrado para esta run.")

st.divider()

# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------

df = load_all_metrics(run_id)

if df.empty:
    st.info("Nenhum resultado disponível ainda. Aguarde os primeiros jobs terminarem.")
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    st.stop()

# Ensure numeric
for col in ["test_rmse", "test_mae", "test_r2", "val_rmse", "val_r2", "train_rmse", "epochs_trained"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------------------------
# Heatmap: Model × Mode (best metric)
# ---------------------------------------------------------------------------

st.subheader("Heatmap — Melhor por Model × Mode")

agg_fn = "min" if ascending else "max"
pivot = (
    df.groupby(["model_name", "mode"])[metric_col]
    .agg(agg_fn)
    .reset_index()
    .pivot(index="model_name", columns="mode", values=metric_col)
)

model_order = ["CNN", "LSTM", "CNN_LSTM", "Transformer", "MLP"]
mode_order  = ["raw", "db4", "learned_wavelet_no_warmup", "learned_wavelet"]
pivot = pivot.reindex(index=model_order, columns=mode_order)

colorscale = "RdYlGn_r" if ascending else "RdYlGn"
fig_heat = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale=colorscale,
    text=[[f"{v:.5f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    showscale=True,
))
fig_heat.update_layout(
    height=300,
    xaxis_title="Mode",
    yaxis_title="Model",
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_heat, width="stretch")

# ---------------------------------------------------------------------------
# Bar chart — best per model+mode
# ---------------------------------------------------------------------------

st.subheader(f"Melhor {metric_col} por combinação")

best = (
    df.groupby(["model_name", "mode"])[metric_col]
    .agg(agg_fn)
    .reset_index()
)
best["label"] = best["model_name"] + " / " + best["mode"]
best = best.sort_values(metric_col, ascending=ascending)

fig_bar = px.bar(
    best, x="label", y=metric_col, color="model_name",
    text=metric_col, text_auto=".5f",
    height=350,
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=80), xaxis_tickangle=-30)
st.plotly_chart(fig_bar, width="stretch")

# ---------------------------------------------------------------------------
# Scatter: train vs test RMSE (overfitting check)
# ---------------------------------------------------------------------------

if "train_rmse" in df.columns and "test_rmse" in df.columns:
    st.subheader("Overfitting check — Train vs Test RMSE")
    fig_sc = px.scatter(
        df, x="train_rmse", y="test_rmse",
        color="model_name", symbol="mode",
        hover_data=["config_idx", "epochs_trained"],
        height=350,
    )
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=df["train_rmse"].max(), y1=df["train_rmse"].max(),
                     line=dict(dash="dash", color="gray"))
    fig_sc.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_sc, width="stretch")

# ---------------------------------------------------------------------------
# Top configs table
# ---------------------------------------------------------------------------

st.subheader("Top configurações")

display_cols = ["model_name", "mode", "config_idx", "test_rmse", "test_mae", "test_r2",
                "val_rmse", "val_r2", "train_rmse", "epochs_trained"]
display_cols = [c for c in display_cols if c in df.columns]

top = df[display_cols].sort_values(metric_col, ascending=ascending).head(30)
st.dataframe(
    top.style.format({c: "{:.5f}" for c in ["test_rmse","test_mae","test_r2","val_rmse","val_r2","train_rmse"] if c in top.columns}),
    width="stretch",
    height=400,
)

# ---------------------------------------------------------------------------
# Running jobs table
# ---------------------------------------------------------------------------

if status:
    running = [j for j in status.get("jobs", []) if j["status"] == "running"]
    if running:
        st.subheader(f"Jobs em execução ({len(running)})")
        run_df = pd.DataFrame([{
            "GPU": j["gpu_id"], "Job": j["name"],
            "Elapsed": f"{int((j['elapsed'] or 0)//60)}m{int((j['elapsed'] or 0)%60)}s",
            "PID": j.get("pid"),
        } for j in running])
        st.dataframe(run_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(10)
    st.rerun()
