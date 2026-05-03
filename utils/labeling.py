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
    num_days: int = 10,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
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
        high: Série opcional de máximas para detectar toques intraday na barreira superior
        low: Série opcional de mínimas para detectar toques intraday na barreira inferior
    
    Returns:
        DataFrame com colunas ['ret', 'label']
    """
    # Seleciona eventos relevantes
    trgt = trgt[trgt > min_ret].dropna()
    if trgt.empty:
        return pd.DataFrame(columns=['ret', 'label'])

    # Define barreiras verticais em dias ÚTEIS (BDay), não corridos.
    # pd.Timedelta(days=N) incluía fins de semana/feriados, fazendo end_time
    # cair fora do índice de preços e descartando silenciosamente ~30% das amostras.
    close_idx = close.index
    idx_positions = {date: pos for pos, date in enumerate(close_idx)}

    t1_list = []
    for date in trgt.index:
        if date not in idx_positions:
            t1_list.append(close_idx[-1])
            continue
        pos = idx_positions[date]
        end_pos = min(pos + num_days, len(close_idx) - 1)
        t1_list.append(close_idx[end_pos])
    t1 = pd.Series(t1_list, index=trgt.index)

    if high is not None:
        high = high.reindex(close.index)
    if low is not None:
        low = low.reindex(close.index)

    # Calcula retornos e aplica barreiras
    out = []
    for idx, end_time in zip(t1.index, t1.values):
        if idx not in idx_positions:
            continue

        price_path = close.loc[idx:end_time]
        if price_path.empty:
            continue

        high_path = high.loc[idx:end_time] if high is not None else None
        low_path = low.loc[idx:end_time] if low is not None else None
        
        start_price = close.loc[idx]
        thresh_up = start_price * (1 + pt_sl[0] * trgt.loc[idx])
        thresh_down = start_price * (1 - pt_sl[1] * trgt.loc[idx])
        
        label = 0
        ret = (price_path.iloc[-1] / start_price) - 1
        t1_actual = price_path.index[-1]
        
        for t, price in price_path.items():
            high_t = high_path.loc[t] if high_path is not None else price
            low_t = low_path.loc[t] if low_path is not None else price

            hit_up = pd.notna(high_t) and high_t >= thresh_up
            hit_down = pd.notna(low_t) and low_t <= thresh_down

            if hit_up and hit_down:
                close_t = price_path.loc[t]
                if close_t > start_price:
                    hit_down = False
                elif close_t < start_price:
                    hit_up = False
                else:
                    hit_up = hit_down = False

            if hit_up:
                label = 1
                ret = (thresh_up / start_price) - 1
                t1_actual = t
                break
            if hit_down:
                label = -1
                ret = (thresh_down / start_price) - 1
                t1_actual = t
                break

        out.append({'index': idx, 'ret': ret, 'label': label, 't1': t1_actual})

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
        num_days=num_days,
        high=df['High'] if 'High' in df.columns else None,
        low=df['Low'] if 'Low' in df.columns else None,
    )
    labels['label'] = labels['label'].astype(int)

    cols = ['label', 'ret'] + (['t1'] if 't1' in labels.columns else [])
    final_df = df.join(labels[cols])
    final_df.dropna(inplace=True)

    return final_df
