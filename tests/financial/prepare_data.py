"""
Prepara dados para todos os ativos: feature engineering + Triple Barrier labeling.

Uso:
    python prepare_data.py              # todos os 25 ativos
    python prepare_data.py PETR4.SA     # apenas um ativo
    python prepare_data.py --force      # reprocessa mesmo que já exista
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))  # project root

from src.data_loader import load_raw, save_processed, PROCESSED_DIR, LABELS_DIR
from src.feature_engineering import compute_features
from src.labeling import label_asset
from config.experiment_config import TICKERS, FEATURE_CONFIG, LABELING_CONFIG


def prepare_ticker(ticker: str, force: bool = False) -> bool:
    proc_path  = PROCESSED_DIR / f"{ticker}.parquet"
    label_path = LABELS_DIR    / f"{ticker}.parquet"

    if not force and proc_path.exists() and label_path.exists():
        logger.info("  SKIP %s (já processado)", ticker)
        return True

    try:
        logger.info("  → %s: carregando dados brutos…", ticker)
        df = load_raw(ticker)

        logger.info("  → %s: computando features (%d linhas)…", ticker, len(df))
        features = compute_features(df, cfg=FEATURE_CONFIG)
        # Substitui ±Inf por NaN antes de dropna: features como obv_roc_10
        # podem gerar Inf por divisão por zero (OBV=0 no denominador).
        features = features.replace([float('inf'), float('-inf')], float('nan'))
        features = features.dropna()

        logger.info("  → %s: aplicando Triple Barrier labeling…", ticker)
        labels, t1 = label_asset(df.loc[features.index], cfg=LABELING_CONFIG)

        # Alinha features e labels pelo índice comum
        common   = features.index.intersection(labels.index)
        features = features.loc[common]
        labels   = labels.loc[common]
        t1       = t1.loc[common]

        save_processed(ticker, features, labels, t1)
        logger.info("  ✓ %s: %d amostras, %d features", ticker, len(features), features.shape[1])
        return True

    except Exception as exc:
        logger.error("  ✗ %s: ERRO — %s", ticker, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepara dados financeiros")
    parser.add_argument("tickers", nargs="*", help="Tickers específicos (padrão: todos)")
    parser.add_argument("--force", action="store_true", help="Reprocessa mesmo que já exista")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else TICKERS
    logger.info("Preparando %d ativos…", len(tickers))

    ok = failed = 0
    for ticker in tickers:
        if prepare_ticker(ticker, force=args.force):
            ok += 1
        else:
            failed += 1

    logger.info("Concluído: %d OK, %d com erro", ok, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
