"""
Módulo de visualização para o experimento SelfRegulationSCP1
(classificação MULTIVARIADA BINÁRIA — 6 canais EEG, negativity/positivity).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Nomes default dos canais EEG (genérico: eeg_0..eeg_5). Usado só para legendas.
_CHANNEL_NAMES = [f"eeg_{i}" for i in range(6)]


class ExperimentVisualizer:
    """Visualizador para análise e apresentação de resultados de classificação multiclasse."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid",
                 figsize: Tuple[int, int] = (12, 6)):
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8")
        self.figsize = figsize
        self.colors = plt.cm.tab10.colors

    # ------------------------------------------------------------------
    # Sinais (multivariados: cada amostra é (length, n_channels))
    # ------------------------------------------------------------------
    def plot_samples(self, X: np.ndarray, y: np.ndarray,
                     n_per_class: int = 3, title: str = "Amostras SelfRegulationSCP1",
                     channel_names: Optional[List[str]] = None,
                     save_path: Optional[Path] = None) -> plt.Figure:
        """Plota amostras de cada classe; cada subplot sobrepõe os canais (x/y/z)."""
        classes = np.unique(y)
        n_channels = X.shape[2] if X.ndim == 3 else 1
        channel_names = channel_names or _CHANNEL_NAMES[:n_channels]
        fig, axes = plt.subplots(len(classes), n_per_class,
                                  figsize=(4 * n_per_class, 2.5 * len(classes)),
                                  sharex=True, sharey=True, squeeze=False)
        for row, cls in enumerate(classes):
            idxs = np.where(y == cls)[0]
            chosen = np.random.choice(idxs, min(n_per_class, len(idxs)), replace=False)
            for col, idx in enumerate(chosen):
                ax = axes[row, col]
                if X.ndim == 3:
                    for c in range(n_channels):
                        ax.plot(X[idx, :, c], linewidth=0.8,
                                label=channel_names[c] if (row == 0 and col == 0) else None)
                else:
                    ax.plot(X[idx], linewidth=0.8)
                ax.set_title(f"Classe {int(cls)} — idx {idx}", fontsize=9)
                ax.grid(True, alpha=0.3)
        if X.ndim == 3:
            axes[0, 0].legend(fontsize=8, loc="upper right")
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_class_distribution(self, y: np.ndarray, title: str = "Distribuição de Classes",
                                save_path: Optional[Path] = None) -> plt.Figure:
        """Barras com a contagem de amostras por classe."""
        classes, counts = np.unique(y, return_counts=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([str(int(c)) for c in classes], counts, color="steelblue")
        ax.set_xlabel("Classe")
        ax.set_ylabel("N amostras")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ------------------------------------------------------------------
    # Wavelet decomposition
    # ------------------------------------------------------------------
    def plot_wavelet_decomposition(self, signal, approx, details,
                                    title="Decomposição Wavelet",
                                    save_path=None):
        n_levels = len(details) + 2
        fig, axes = plt.subplots(n_levels, 1, figsize=(14, 2.5 * n_levels))
        axes[0].plot(signal, linewidth=0.5)
        axes[0].set_title("Sinal Original", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(approx, linewidth=0.5, color='green')
        axes[1].set_title("Aproximação (Low-pass)", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        for i, d in enumerate(details):
            axes[i + 2].plot(d, linewidth=0.5, color='orange')
            axes[i + 2].set_title(f"Detalhe Nível {i+1} (High-pass)", fontsize=11)
            axes[i + 2].grid(True, alpha=0.3)
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ------------------------------------------------------------------
    # Classificação
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model",
                               class_names=None, normalize=False, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        class_names = class_names or [str(c) for c in sorted(np.unique(y_true))]
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão — {model_name}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_roc_ovr(self, y_true, y_proba, n_classes: int, model_name="Model",
                     save_path=None):
        """Curvas ROC one-vs-rest (uma por classe) para classificação multiclasse."""
        classes = list(range(n_classes))
        y_bin = label_binarize(y_true, classes=classes)
        fig, ax = plt.subplots(figsize=(8, 7))
        for c in classes:
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
                ax.plot(fpr, tpr, linewidth=1.5, label=f"Classe {c} (AUC={auc(fpr, tpr):.3f})")
            except Exception:
                continue
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC One-vs-Rest — {model_name}")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ------------------------------------------------------------------
    # Comparação de modelos
    # ------------------------------------------------------------------
    def plot_model_comparison(self, results_df, metrics=None, save_path=None):
        metrics = metrics or ["accuracy", "f1_macro", "auc_ovr"]
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
        if n == 1:
            axes = [axes]
        for idx, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue
            data = results_df[metric].sort_values(ascending=True)
            colors = ['green' if v == data.max() else 'steelblue' for v in data.values]
            bars = axes[idx].barh(data.index, data.values, color=colors)
            axes[idx].set_xlabel(metric.upper())
            axes[idx].set_title(f"Comparação: {metric.upper()}")
            axes[idx].grid(True, alpha=0.3, axis='x')
            for bar, val in zip(bars, data.values):
                axes[idx].text(val, bar.get_y() + bar.get_height() / 2,
                               f'{val:.4f}', va='center', ha='left', fontsize=9)
        plt.suptitle("Comparação de Modelos", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------
    def plot_training_history(self, history, title="Training History", save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Train Acc')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def plot_feature_importance(self, feature_names, importances, top_n=20,
                                title="Feature Importance", save_path=None):
        indices = np.argsort(importances)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importances[indices], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importância')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
