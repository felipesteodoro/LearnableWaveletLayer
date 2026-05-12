"""
Módulo de visualização para experimentos.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd


class ExperimentVisualizer:
    """
    Visualizador para análise e apresentação de resultados.
    """
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: Tuple[int, int] = (12, 6)):
        """
        Args:
            style: Estilo do matplotlib
            figsize: Tamanho padrão das figuras
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("seaborn-v0_8")
        self.figsize = figsize
        self.colors = plt.cm.tab10.colors
        
    def plot_signal(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        title: str = "Sinal Sintético",
        start: int = 0,
        length: int = 2000,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota sinal ruidoso vs limpo.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        t = np.arange(start, min(start + length, len(noisy)))
        noisy_segment = noisy[t]
        clean_segment = clean[t]
        
        # Sinal ruidoso
        axes[0].plot(t, noisy_segment, alpha=0.7, linewidth=0.5, color='blue')
        axes[0].set_title("Sinal Ruidoso (Input)", fontsize=12)
        axes[0].set_xlabel("Tempo")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        # Sinal limpo
        axes[1].plot(t, clean_segment, alpha=0.9, linewidth=0.8, color='green')
        axes[1].set_title("Sinal Limpo (Target)", fontsize=12)
        axes[1].set_xlabel("Tempo")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        
        # Sobreposição
        axes[2].plot(t, noisy_segment, alpha=0.5, linewidth=0.5, label='Ruidoso', color='blue')
        axes[2].plot(t, clean_segment, alpha=0.9, linewidth=0.8, label='Limpo', color='green')
        axes[2].set_title("Comparação", fontsize=12)
        axes[2].set_xlabel("Tempo")
        axes[2].set_ylabel("Amplitude")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_components(
        self,
        components: Dict[str, np.ndarray],
        length: int = 2000,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota componentes individuais do sinal.
        """
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=(14, 3 * n_components))
        
        if n_components == 1:
            axes = [axes]
        
        t = np.arange(min(length, len(list(components.values())[0])))
        
        for idx, (name, component) in enumerate(components.items()):
            axes[idx].plot(t, component[:len(t)], linewidth=0.7)
            axes[idx].set_title(f"Componente: {name}", fontsize=11)
            axes[idx].set_xlabel("Tempo")
            axes[idx].set_ylabel("Amplitude")
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle("Componentes do Sinal Sintético", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_wavelet_decomposition(
        self,
        signal: np.ndarray,
        approx: np.ndarray,
        details: List[np.ndarray],
        title: str = "Decomposição Wavelet",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota decomposição wavelet multi-nível.
        """
        n_levels = len(details) + 2  # sinal + approx + details
        fig, axes = plt.subplots(n_levels, 1, figsize=(14, 2.5 * n_levels))
        
        # Sinal original
        axes[0].plot(signal[:2000], linewidth=0.5)
        axes[0].set_title("Sinal Original", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Aproximação
        axes[1].plot(approx[:1000], linewidth=0.5, color='green')
        axes[1].set_title("Aproximação (Low-pass)", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Detalhes
        for i, detail in enumerate(details):
            axes[i + 2].plot(detail[:1000], linewidth=0.5, color='orange')
            axes[i + 2].set_title(f"Detalhe Nível {i + 1} (High-pass)", fontsize=11)
            axes[i + 2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        n_samples: int = 500,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota comparação entre valores reais e preditos.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Série temporal
        t = np.arange(min(n_samples, len(y_true)))
        axes[0, 0].plot(t, y_true[:len(t)], label='Real', alpha=0.7, linewidth=0.8)
        axes[0, 0].plot(t, y_pred[:len(t)], label='Predito', alpha=0.7, linewidth=0.8)
        axes[0, 0].set_title("Série Temporal: Real vs Predito")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.3, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 1].set_xlabel("Real")
        axes[0, 1].set_ylabel("Predito")
        axes[0, 1].set_title("Scatter: Real vs Predito")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Resíduos
        residuals = y_true - y_pred
        axes[1, 0].plot(residuals[:len(t)], linewidth=0.5, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title("Resíduos ao Longo do Tempo")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histograma dos resíduos
        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_title("Distribuição dos Resíduos")
        axes[1, 1].set_xlabel("Resíduo")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Análise de Predição - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = ["rmse", "mae", "r2"],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota comparação entre modelos.
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in results_df.columns:
                data = results_df[metric].sort_values()
                colors = ['green' if v == data.min() else 'steelblue' for v in data.values]
                
                bars = axes[idx].barh(data.index, data.values, color=colors)
                axes[idx].set_xlabel(metric.upper())
                axes[idx].set_title(f"Comparação: {metric.upper()}")
                axes[idx].grid(True, alpha=0.3, axis='x')
                
                # Adicionar valores
                for bar, val in zip(bars, data.values):
                    axes[idx].text(val, bar.get_y() + bar.get_height()/2, 
                                   f'{val:.4f}', va='center', ha='left', fontsize=9)
        
        plt.suptitle("Comparação de Modelos", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota histórico de treinamento de modelo deep learning.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss ao Longo do Treinamento')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Métricas adicionais
        other_metrics = [k for k in history.keys() if k not in ['loss', 'val_loss']]
        for metric in other_metrics[:4]:  # Limitar a 4 métricas
            if not metric.startswith('val_'):
                axes[1].plot(history[metric], label=metric)
                if f'val_{metric}' in history:
                    axes[1].plot(history[f'val_{metric}'], label=f'val_{metric}', linestyle='--')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Métricas ao Longo do Treinamento')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota importância de features.
        """
        # Ordenar por importância
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
    
    def plot_cv_results(
        self,
        cv_results: Dict[str, Dict],
        metrics: List[str] = ["rmse", "mae", "r2"],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plota resultados de cross-validation.
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in cv_results:
                data = cv_results[metric]
                per_fold = data.get('per_fold', [])
                mean = data.get('mean', 0)
                std = data.get('std', 0)
                
                if per_fold:
                    x = np.arange(1, len(per_fold) + 1)
                    axes[idx].bar(x, per_fold, alpha=0.7, label='Per Fold')
                    axes[idx].axhline(y=mean, color='r', linestyle='--', 
                                       label=f'Mean: {mean:.4f} ± {std:.4f}')
                    axes[idx].fill_between([0.5, len(per_fold) + 0.5], 
                                           mean - std, mean + std, alpha=0.2, color='r')
                    axes[idx].set_xlabel('Fold')
                    axes[idx].set_ylabel(metric.upper())
                    axes[idx].set_title(f'CV Results: {metric.upper()}')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle("Resultados Cross-Validation", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def create_summary_report(
    results_df: pd.DataFrame,
    save_path: Path,
    title: str = "Relatório de Experimentos"
):
    """
    Cria relatório HTML com resumo dos resultados.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
            .best {{ background-color: #90EE90; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Data de geração: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Resumo dos Resultados</h2>
        {results_df.to_html(classes='dataframe')}
        
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)
