"""
Módulo para avaliação de performance do modelo e visualizações.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def calculate_financial_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calcula métricas financeiras para avaliação de estratégia.
    
    Args:
        returns: Série de retornos da estratégia
        risk_free_rate: Taxa livre de risco (anual)
    
    Returns:
        Dicionário com métricas:
        - Sharpe Ratio: Retorno ajustado ao risco
        - Sortino Ratio: Retorno ajustado ao risco de queda
        - Maximum Drawdown: Máxima perda do pico ao vale
        - Calmar Ratio: Retorno anualizado / |MDD|
        - Cumulative Returns: Série de retornos acumulados
    """
    returns = pd.Series(returns).dropna()
    
    if returns.empty or returns.std() == 0:
        return {
            'Sharpe Ratio': 0.0, 
            'Sortino Ratio': 0.0, 
            'Maximum Drawdown': 0.0, 
            'Calmar Ratio': 0.0, 
            'Cumulative Returns': pd.Series([1.0])
        }
    
    # Sharpe Ratio (anualizado)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    # Sortino Ratio
    downside_std = returns[returns < 0].std()
    sortino = (returns.mean() * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0
    
    # Maximum Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret - peak) / peak
    mdd = drawdown.min()
    
    # Calmar Ratio
    calmar = (returns.mean() * 252) / abs(mdd) if mdd != 0 else 0
    
    return {
        'Sharpe Ratio': sharpe, 
        'Sortino Ratio': sortino, 
        'Maximum Drawdown': mdd, 
        'Calmar Ratio': calmar, 
        'Cumulative Returns': cum_ret
    }


def plot_labels_on_price(
    df: pd.DataFrame, 
    ticker: str, 
    n_points: int = 500,
    figsize: tuple = (18, 9)
) -> None:
    """
    Visualiza os rótulos de trading na série de preços.
    
    Args:
        df: DataFrame com colunas 'Close' e 'label'
        ticker: Nome do ativo para título
        n_points: Número de pontos a plotar (últimos n)
        figsize: Tamanho da figura
    """
    df_subset = df.iloc[-n_points:]
    close_price = df_subset['Close']
    
    buy_signals = df_subset[df_subset['label'] == 1]
    sell_signals = df_subset[df_subset['label'] == -1]
    hold_signals = df_subset[df_subset['label'] == 0]
    
    plt.figure(figsize=figsize)
    plt.plot(close_price.index, close_price.values, label='Preço de Fechamento', color='k', alpha=0.7)
    
    plt.scatter(
        buy_signals.index, 
        close_price.loc[buy_signals.index], 
        label='Compra (1)', 
        marker='^', 
        color='green', 
        s=100, 
        alpha=0.9, 
        zorder=5
    )
    plt.scatter(
        sell_signals.index, 
        close_price.loc[sell_signals.index], 
        label='Venda (-1)', 
        marker='v', 
        color='red', 
        s=100, 
        alpha=0.9, 
        zorder=5
    )
    plt.scatter(
        hold_signals.index, 
        close_price.loc[hold_signals.index], 
        label='Manter (0)', 
        marker='o', 
        color='blue', 
        s=20, 
        alpha=0.6, 
        zorder=4
    )
    
    plt.title(f'Sinais da Barreira Tripla vs. Preço de Fechamento - {ticker} ({n_points} dias)')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    title: str = 'Matriz de Confusão',
    figsize: tuple = (8, 6)
) -> None:
    """
    Plota matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels previstos
        title: Título do gráfico
        figsize: Tamanho da figura
    """
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Vender', 'Manter', 'Comprar'], 
        yticklabels=['Vender', 'Manter', 'Comprar']
    )
    plt.title(title)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()


def plot_cumulative_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    ticker: str = '',
    title_suffix: str = '',
    figsize: tuple = (12, 6)
) -> None:
    """
    Plota retornos acumulados da estratégia vs benchmark.
    
    Args:
        strategy_returns: Série de retornos da estratégia
        benchmark_returns: Série de retornos do benchmark (opcional)
        ticker: Nome do ativo
        title_suffix: Sufixo para o título (ex: 'OOS', 'CV')
        figsize: Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    strategy_cum = (1 + strategy_returns).cumprod()
    strategy_cum.plot(label='Estratégia')
    
    if benchmark_returns is not None:
        benchmark_cum = (1 + benchmark_returns).cumprod()
        benchmark_cum.plot(label='Buy and Hold', linestyle='--', linewidth=2)
    
    title = f'Retorno Acumulado da Estratégia'
    if title_suffix:
        title += f' ({title_suffix})'
    if ticker:
        title += f' - {ticker}'
    
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Retorno Acumulado')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    title: str = ''
) -> None:
    """
    Imprime relatório de classificação formatado.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels previstos
        title: Título do relatório
    """
    if title:
        print(f"\n--- {title} ---")
    
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Relatório de Classificação:")
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=['Vender (-1)', 'Manter (0)', 'Comprar (1)'], 
        labels=[-1, 0, 1], 
        zero_division=0
    ))


def print_financial_metrics(metrics: dict, title: str = '') -> None:
    """
    Imprime métricas financeiras formatadas.
    
    Args:
        metrics: Dicionário retornado por calculate_financial_metrics
        title: Título do relatório
    """
    if title:
        print(f"\n--- {title} ---")
    
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['Maximum Drawdown']:.4%}")
    print(f"Calmar Ratio: {metrics['Calmar Ratio']:.4f}")
