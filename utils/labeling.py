"""
Módulo para rotulagem de dados financeiros usando Triple Barrier Method.
"""
import pandas as pd
import numpy as np


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """
    Calcula a volatilidade diária dos retornos usando médias móveis exponenciais.
    
    Args:
        close: Série de preços de fechamento
        span: Período para a média móvel exponencial
    
    Returns:
        Série com a volatilidade diária
    """
    df0 = close.pct_change()
    df0 = df0.ewm(span=span).std()
    df0.dropna(inplace=True)
    return df0


def triple_barrier_labeling(
    close: pd.Series, 
    trgt: pd.Series, 
    pt_sl: list = [1.5, 1.5], 
    min_ret: float = 0.001, 
    num_days: int = 10
) -> pd.DataFrame:
    """
    Rotulagem de barreira tripla para séries temporais financeiras.
    
    O método define três barreiras:
    - Superior (take profit): preço inicial * (1 + pt * volatilidade)
    - Inferior (stop loss): preço inicial * (1 - sl * volatilidade)
    - Vertical (tempo): após num_days dias
    
    Labels:
    - 1: Take profit atingido primeiro (compra)
    - -1: Stop loss atingido primeiro (venda)
    - 0: Barreira vertical atingida (manter)
    
    Args:
        close: Série de preços de fechamento (index datetime)
        trgt: Série de volatilidade alvo (index datetime)
        pt_sl: [pt, sl] multiplicadores para take profit e stop loss
        min_ret: Retorno mínimo para considerar evento
        num_days: Horizonte da barreira vertical em dias
    
    Returns:
        DataFrame com colunas ['ret', 'label']
    """
    # Seleciona eventos relevantes
    trgt = trgt[trgt > min_ret].dropna()
    if trgt.empty:
        return pd.DataFrame(columns=['ret', 'label'])

    # Define barreiras verticais
    t1 = trgt.index + pd.Timedelta(days=num_days)
    t1 = t1.where(t1 < close.index[-1], close.index[-1])
    t1 = pd.Series(t1, index=trgt.index)

    # Calcula retornos e aplica barreiras
    out = []
    for idx, end_time in zip(t1.index, t1.values):
        if idx not in close.index or end_time not in close.index:
            continue
        
        price_path = close.loc[idx:end_time]
        if price_path.empty:
            continue
        
        start_price = close.loc[idx]
        thresh_up = start_price * (1 + pt_sl[0] * trgt.loc[idx])
        thresh_down = start_price * (1 - pt_sl[1] * trgt.loc[idx])
        
        label = 0
        ret = (price_path.iloc[-1] / start_price) - 1
        
        for t, price in price_path.items():
            if price >= thresh_up:
                label = 1
                ret = (price / start_price) - 1
                break
            elif price <= thresh_down:
                label = -1
                ret = (price / start_price) - 1
                break
        
        out.append({'index': idx, 'ret': ret, 'label': label})

    df_out = pd.DataFrame(out).set_index('index')
    return df_out


def apply_labeling(df: pd.DataFrame, pt_sl: list = [1.5, 1.5], min_ret: float = 0.001, num_days: int = 10) -> pd.DataFrame:
    """
    Aplica a rotulagem triple barrier a um DataFrame completo.
    
    Args:
        df: DataFrame com coluna 'Close'
        pt_sl: Multiplicadores para take profit e stop loss
        min_ret: Retorno mínimo para considerar evento
        num_days: Horizonte da barreira vertical
    
    Returns:
        DataFrame original com colunas 'label', 'ret' e 'trgt' adicionadas
    """
    close_prices = df['Close']
    daily_vol = get_daily_vol(close_prices)
    df['trgt'] = daily_vol

    labels = triple_barrier_labeling(
        close=df['Close'],
        trgt=df['trgt'],
        pt_sl=pt_sl,
        min_ret=min_ret,
        num_days=num_days
    )
    labels['label'] = labels['label'].astype(int)
    
    final_df = df.join(labels[['label', 'ret']])
    final_df.dropna(inplace=True)
    
    return final_df
