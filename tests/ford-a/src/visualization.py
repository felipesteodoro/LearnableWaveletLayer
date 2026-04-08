"""
Módulo de visualização para experimentos de classificação FordA.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ExperimentVisualizer:
    """Visualizador para análise e apresentação de resultados de classificação."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid",
                 figsize: Tuple[int, int] = (12, 6)):
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8")
        self.figsize = figsize
        self.colors = plt.cm.tab10.colors

    # ------------------------------------------------------------------
    # Sinais
    # ------------------------------------------------------------------
    def plot_samples(self, X: np.ndarray, y: np.ndarray,
                     n_per_class: int = 4, title: str = "Amostras FordA",
                     save_path: Optional[Path] = None) -> plt.Figure:
        """Plota amostras aleatórias de cada classe."""
        classes = np.unique(y)
        fig, axes = plt.subplots(len(classes), n_per_class,
                                  figsize=(4 * n_per_class, 3 * len(classes)),
                                  sharex=True, sharey=True)
        for row, cls in enumerate(classes):
            idxs = np.where(y == cls)[0]
            chosen = np.random.choice(idxs, min(n_per_class, len(idxs)), replace=False)
            for col, idx in enumerate(chosen):
                ax = axes[row, col] if len(classes) > 1 else axes[col]
                ax.plot(X[idx], linewidth=0.7)
                ax.set_title(f"Classe {int(cls)} — idx {idx}", fontsize=9)
                ax.grid(True, alpha=0.3)
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_class_overlay(self, X: np.ndarray, y: np.ndarray,
                           n_per_class: int = 30,
                           title: str = "Sobreposição por Classe",
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Sobreposição de sinais por classe."""
        classes = np.unique(y)
        fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 4))
        if len(classes) == 1:
            axes = [axes]
        for i, cls in enumerate(classes):
            idxs = np.where(y == cls)[0]
            chosen = np.random.choice(idxs, min(n_per_class, len(idxs)), replace=False)
            for idx in chosen:
                axes[i].plot(X[idx], alpha=0.3, linewidth=0.5)
            axes[i].plot(X[idxs].mean(axis=0), color='black', linewidth=2, label='Média')
            axes[i].set_title(f"Classe {int(cls)} ({len(idxs)} amostras)")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        plt.suptitle(title, fontsize=14, fontweight='bold')
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
                               class_names=None, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        class_names = class_names or [str(c) for c in sorted(np.unique(y_true))]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão — {model_name}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_roc_curve(self, y_true, y_prob, model_name="Model", save_path=None):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_multi_roc(self, roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                       save_path=None):
        """roc_data: {name: (fpr, tpr, auc_score)}"""
        fig, ax = plt.subplots(figsize=(8, 7))
        for name, (fpr, tpr, auc_val) in roc_data.items():
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_val:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curves — Comparação")
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ------------------------------------------------------------------
    # Comparação de modelos
    # ------------------------------------------------------------------
    def plot_model_comparison(self, results_df, metrics=None, save_path=None):
        metrics = metrics or ["accuracy", "f1", "auc_roc"]
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
