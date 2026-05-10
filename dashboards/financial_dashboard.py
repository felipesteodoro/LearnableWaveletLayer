"""
Streamlit dashboard — Financial Experiment (multiclass classification)

Run:
    streamlit run dashboards/financial_dashboard.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

FINANCIAL_DIR = Path(__file__).parent.parent / "tests" / "financial"
RESULTS_BASE  = FINANCIAL_DIR / "results"

st.set_page_config(
    page_title="Financial Experiment",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def load_queue_status(run_id: str) -> dict | None:
    """Tenta queue_status.json diretamente na pasta da run."""
    path = RESULTS_BASE / run_id / "queue_status.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_all_metrics(run_id: str) -> pd.DataFrame:
    """
    Lê todos os metrics.json dentro da run.
    Estrutura esperada: results/{run_id}/{feature_mode}/{ticker}/{model}_{mode}/metrics.json
    """
    rows = []
    base = RESULTS_BASE / run_id
    for f in base.rglob("metrics.json"):
        try:
            m = json.loads(f.read_text())
            # Infer feature_mode and ticker from path if not in JSON
            parts = f.parts
            try:
                run_idx  = parts.index(run_id)
                feature_mode = parts[run_idx + 1]
                ticker       = parts[run_idx + 2]
                model_mode   = parts[run_idx + 3]
            except (ValueError, IndexError):
                feature_mode = m.get("feature_mode", "?")
                ticker       = m.get("ticker", "?")
                model_mode   = "?"
            m.setdefault("feature_mode", feature_mode)
            m.setdefault("ticker", ticker)
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
    return sorted(
        [d.name for d in RESULTS_BASE.iterdir()
         if d.is_dir() and (d / "queue_status.json").exists()],
        reverse=True,
    )


def load_exp_config() -> dict:
    try:
        import sys
        cfg_path = str(FINANCIAL_DIR / "config")
        if cfg_path not in sys.path:
            sys.path.insert(0, cfg_path)
        from experiment_config import DL_MODELS_CONFIG, DL_TRAINING_CONFIG, LEARNED_WAVELET_CONFIG
        return {"models": DL_MODELS_CONFIG, "training": DL_TRAINING_CONFIG, "wavelet": LEARNED_WAVELET_CONFIG}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📈 Financial Experiment")
runs = list_runs()

if not runs:
    st.warning("Nenhuma run encontrada. Execute `run_dl_queue.py` em `tests/financial/` primeiro.")
    st.stop()

run_id       = st.sidebar.selectbox("Run ID", runs, index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)

# Metric selector
fin_metric = st.sidebar.selectbox(
    "Métrica financeira principal",
    ["fin_metrics.oos_sharpe", "fin_metrics.oos_accuracy",
     "fin_metrics.oos_bh_sharpe", "fin_metrics.oos_sortino"],
    index=0,
)
ml_metric = st.sidebar.selectbox(
    "Métrica ML principal",
    ["ml_metrics.accuracy", "ml_metrics.f1_macro", "ml_metrics.roc_auc_ovr"],
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

st.title("📈 Financial Experiment Dashboard")
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

# Flatten nested fin/ml metrics if needed
for key in ["fin_metrics.oos_sharpe","fin_metrics.oos_accuracy","fin_metrics.oos_bh_sharpe",
            "fin_metrics.oos_sortino","fin_metrics.oos_max_drawdown","fin_metrics.oos_cagr",
            "ml_metrics.accuracy","ml_metrics.f1_macro","ml_metrics.roc_auc_ovr"]:
    if key in df.columns:
        df[key] = pd.to_numeric(df[key], errors="coerce")

# Filter controls
col_fmode, col_model = st.columns(2)
all_fmodes = sorted(df["feature_mode"].dropna().unique().tolist()) if "feature_mode" in df.columns else []
all_models = sorted(df["model_name"].dropna().unique().tolist()) if "model_name" in df.columns else []
all_modes  = sorted(df["mode"].dropna().unique().tolist()) if "mode" in df.columns else []

sel_fmodes = col_fmode.multiselect("Feature Mode", all_fmodes, default=all_fmodes)
sel_models = col_model.multiselect("Modelos", all_models, default=all_models)

mask = pd.Series([True] * len(df))
if sel_fmodes and "feature_mode" in df.columns:
    mask &= df["feature_mode"].isin(sel_fmodes)
if sel_models and "model_name" in df.columns:
    mask &= df["model_name"].isin(sel_models)
df_f = df[mask].copy()

st.caption(f"{len(df_f)} resultados filtrados de {len(df)} totais")

# ---------------------------------------------------------------------------
# Heatmap: Model × Mode (OOS Sharpe avg over tickers)
# ---------------------------------------------------------------------------

if fin_metric in df_f.columns and "model_name" in df_f.columns and "mode" in df_f.columns:
    st.subheader(f"Heatmap — {fin_metric} médio por Model × Mode")

    pivot = (
        df_f.groupby(["model_name", "mode"])[fin_metric]
        .mean()
        .reset_index()
        .pivot(index="model_name", columns="mode", values=fin_metric)
    )
    model_order = ["CNN", "LSTM", "CNN_LSTM", "MLP", "Transformer"]
    mode_order  = ["raw", "db4", "learned_wavelet", "learned_wavelet_no_warmup"]
    pivot = pivot.reindex(
        index=[m for m in model_order if m in pivot.index],
        columns=[c for c in mode_order if c in pivot.columns],
    )

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max() if pivot.size else 1, 0.01)
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        text=np.round(pivot.values, 3).astype(str),
        texttemplate="%{text}",
        zmid=0,
        zmin=-vmax, zmax=vmax,
        showscale=True,
    ))
    fig_heat.update_layout(height=300, xaxis_title="Mode", yaxis_title="Model",
                           margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Heatmap: Ticker × Model (OOS Sharpe)
# ---------------------------------------------------------------------------

if fin_metric in df_f.columns and "ticker" in df_f.columns and "model_name" in df_f.columns:
    st.subheader(f"Heatmap — {fin_metric} por Ticker × Model (melhor mode por célula)")

    pivot2 = (
        df_f.groupby(["ticker", "model_name"])[fin_metric]
        .mean()
        .reset_index()
        .pivot(index="ticker", columns="model_name", values=fin_metric)
    )
    vmax2 = max(abs(pivot2.values[~np.isnan(pivot2.values)]).max() if pivot2.size else 1, 0.01)
    fig_heat2 = go.Figure(go.Heatmap(
        z=pivot2.values,
        x=pivot2.columns.tolist(),
        y=pivot2.index.tolist(),
        colorscale="RdYlGn",
        text=np.round(pivot2.values, 2).astype(str),
        texttemplate="%{text}",
        zmid=0, zmin=-vmax2, zmax=vmax2,
        showscale=True,
    ))
    fig_heat2.update_layout(height=max(300, len(pivot2.index) * 22),
                            xaxis_title="Model", yaxis_title="Ticker",
                            margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_heat2, use_container_width=True)

# ---------------------------------------------------------------------------
# Sharpe distribution by mode
# ---------------------------------------------------------------------------

if fin_metric in df_f.columns and "mode" in df_f.columns:
    st.subheader(f"Distribuição de {fin_metric} por Mode")
    fig_box = px.box(
        df_f.dropna(subset=[fin_metric]),
        x="mode", y=fin_metric, color="model_name",
        points="outliers", height=380,
    )
    fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_box.update_layout(margin=dict(l=10,r=10,t=30,b=60), xaxis_tickangle=-20)
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------------------------
# Scatter: OOS Accuracy vs OOS Sharpe
# ---------------------------------------------------------------------------

if "fin_metrics.oos_sharpe" in df_f.columns and "fin_metrics.oos_accuracy" in df_f.columns:
    st.subheader("OOS Accuracy vs OOS Sharpe")
    fig_sc = px.scatter(
        df_f.dropna(subset=["fin_metrics.oos_sharpe","fin_metrics.oos_accuracy"]),
        x="fin_metrics.oos_accuracy", y="fin_metrics.oos_sharpe",
        color="model_name", symbol="mode",
        hover_data=["ticker"] if "ticker" in df_f.columns else [],
        height=380,
    )
    fig_sc.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sc.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

# ---------------------------------------------------------------------------
# Top results table
# ---------------------------------------------------------------------------

st.subheader("Top resultados")

table_cols = ["ticker","model_name","mode","feature_mode",
              "fin_metrics.oos_sharpe","fin_metrics.oos_accuracy",
              "fin_metrics.oos_bh_sharpe","fin_metrics.oos_sortino",
              "fin_metrics.oos_max_drawdown","ml_metrics.accuracy","ml_metrics.f1_macro"]
table_cols = [c for c in table_cols if c in df_f.columns]

if fin_metric in df_f.columns:
    top = df_f[table_cols].sort_values(fin_metric, ascending=False).head(40)
    fmt = {c: "{:.4f}" for c in table_cols if c not in ["ticker","model_name","mode","feature_mode"]}
    st.dataframe(top.style.format(fmt), use_container_width=True, height=450)

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
