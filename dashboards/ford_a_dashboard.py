"""
Streamlit dashboard — Ford-A Experiment (binary classification)

Run:
    streamlit run dashboards/ford_a_dashboard.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

FORDA_DIR    = Path(__file__).parent.parent / "tests" / "ford-a"
RESULTS_BASE = FORDA_DIR / "results"

st.set_page_config(
    page_title="Ford-A Experiment",
    page_icon="🚗",
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
    for f in (RESULTS_BASE / run_id).rglob("metrics.json"):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def list_runs() -> list[str]:
    if not RESULTS_BASE.exists():
        return []
    return sorted([d.name for d in RESULTS_BASE.iterdir()
                   if d.is_dir() and d.name[0].isdigit()
                   and (d / "queue_status.json").exists()], reverse=True)


def load_exp_config() -> dict:
    try:
        import sys
        cfg_path = str(FORDA_DIR / "config")
        if cfg_path not in sys.path:
            sys.path.insert(0, cfg_path)
        from experiment_config import FORDA_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG
        return {"dataset": FORDA_CONFIG, "training": DL_TRAINING_CONFIG, "wavelet": LEARNED_WAVELET_CONFIG}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🚗 Ford-A Experiment")
runs = list_runs()

if not runs:
    st.warning("Nenhuma run encontrada. Execute `run_dl_queue.py` em `tests/ford-a/` primeiro.")
    st.stop()

run_id       = st.sidebar.selectbox("Run ID", runs, index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)
metric_col   = st.sidebar.selectbox(
    "Métrica principal",
    ["test_accuracy", "test_f1_macro", "test_auc_roc", "test_f1_weighted"],
    index=0,
)

cfg = load_exp_config()
if cfg:
    with st.sidebar.expander("Configuração do experimento", expanded=False):
        st.json(cfg)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

status = load_queue_status(run_id)

st.title("🚗 Ford-A Experiment Dashboard")
st.caption(f"Run: **{run_id}**  |  {pd.Timestamp.now().strftime('%H:%M:%S')}")

if status:
    s = status["summary"]
    total = max(s["total"], 1)
    pct   = s["done"] / total

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total",     s["total"])
    c2.metric("✅ Done",    s["done"])
    c3.metric("▶ Running",  s["running"])
    c4.metric("↻ Retry",    s["retrying"])
    c5.metric("❌ Failed",   s["failed"])

    st.progress(pct, text=f"{pct*100:.1f}% concluído")

    elapsed = time.time() - status.get("queue_start_time", time.time())
    eta_str = "—"
    if s["done"] > 0 and (s["pending"] + s["retrying"]) > 0:
        rate = s["done"] / elapsed
        eta_sec = (s["pending"] + s["retrying"]) / rate
        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
    st.caption(f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s   |   ETA: {eta_str}")

st.divider()

# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------

df = load_all_metrics(run_id)

if df.empty:
    st.info("Nenhum resultado disponível ainda.")
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    st.stop()

for col in ["test_accuracy","test_f1_macro","test_auc_roc","test_f1_weighted",
            "val_accuracy","val_f1_macro","train_accuracy","epochs_trained"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

st.subheader("Heatmap — Melhor por Model × Mode")

pivot = (
    df.groupby(["model_name", "mode"])[metric_col]
    .max()
    .reset_index()
    .pivot(index="model_name", columns="mode", values=metric_col)
)

model_order = ["CNN", "LSTM", "CNN_LSTM", "Transformer"]
mode_order  = ["raw", "db4", "learned_wavelet_no_warmup", "learned_wavelet"]
pivot = pivot.reindex(
    index=[m for m in model_order if m in pivot.index],
    columns=[c for c in mode_order if c in pivot.columns],
)

fig_heat = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale="RdYlGn",
    text=np.round(pivot.values, 4).astype(str),
    texttemplate="%{text}",
    showscale=True,
    zmin=0.5, zmax=1.0,
))
fig_heat.update_layout(height=300, xaxis_title="Mode", yaxis_title="Model",
                       margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Multi-metric bar chart
# ---------------------------------------------------------------------------

st.subheader("Comparação de métricas por combinação")

multi_metrics = [c for c in ["test_accuracy","test_f1_macro","test_auc_roc"] if c in df.columns]
best = df.groupby(["model_name", "mode"])[multi_metrics].max().reset_index()
best["label"] = best["model_name"] + " / " + best["mode"]
best_melted = best.melt(id_vars=["label","model_name","mode"], value_vars=multi_metrics,
                         var_name="metric", value_name="value")

fig_bar = px.bar(
    best_melted, x="label", y="value", color="metric",
    barmode="group", text_auto=".3f", height=380,
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(margin=dict(l=10,r=10,t=30,b=80), xaxis_tickangle=-30,
                      yaxis=dict(range=[0, 1.05]))
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Scatter: Accuracy vs AUC
# ---------------------------------------------------------------------------

if "test_accuracy" in df.columns and "test_auc_roc" in df.columns:
    st.subheader("Accuracy vs AUC-ROC")
    fig_sc = px.scatter(
        df, x="test_accuracy", y="test_auc_roc",
        color="model_name", symbol="mode",
        hover_data=["config_idx","epochs_trained"],
        height=350,
    )
    fig_sc.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

# ---------------------------------------------------------------------------
# Top configs table
# ---------------------------------------------------------------------------

st.subheader("Top configurações")

display_cols = ["model_name","mode","config_idx","test_accuracy","test_f1_macro",
                "test_auc_roc","val_accuracy","val_f1_macro","epochs_trained"]
display_cols = [c for c in display_cols if c in df.columns]

top = df[display_cols].sort_values(metric_col, ascending=False).head(30)
fmt = {c: "{:.4f}" for c in display_cols if c not in ["model_name","mode","config_idx","epochs_trained"]}
st.dataframe(top.style.format(fmt), use_container_width=True, height=400)

# ---------------------------------------------------------------------------
# Running jobs
# ---------------------------------------------------------------------------

if status:
    running = [j for j in status.get("jobs", []) if j["status"] == "running"]
    if running:
        st.subheader(f"Jobs em execução ({len(running)})")
        run_df = pd.DataFrame([{
            "GPU": j["gpu_id"], "Job": j["name"],
            "Elapsed": f"{int((j['elapsed'] or 0)//60)}m{int((j['elapsed'] or 0)%60)}s",
        } for j in running])
        st.dataframe(run_df, use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(10)
    st.rerun()
