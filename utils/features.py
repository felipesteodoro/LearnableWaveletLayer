"""
Módulo para criação de features técnicas e de processamento de sinal.
"""
import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft, irfft


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features técnicas e de processamento de sinal.
    
    Inclui:
    - Indicadores técnicos: SMA, EMA, RSI, MACD, Bandas de Bollinger
    - Features de processamento de sinal: FFT smoothed, Wavelet denoised
    
    Args:
        df: DataFrame com colunas ['Close', 'High', 'Low', 'Volume']
    
    Returns:
        DataFrame com todas as features adicionadas
    
    Raises:
        ValueError: Se o DataFrame estiver vazio ou faltar colunas necessárias
    """
    if df.empty:
        raise ValueError("O DataFrame está vazio.")
    
    required_cols = ['Close', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada. Disponíveis: {df.columns.tolist()}")

    df = df.sort_index()
    features_df = pd.DataFrame(index=df.index)
    close = df['Close']

    # --- Indicadores Técnicos ---
    features_df = _add_technical_indicators(features_df, close)
    
    # --- Features de Processamento de Sinal ---
    features_df = _add_signal_features(features_df, close, df)

    return pd.concat([df, features_df], axis=1)


def _add_technical_indicators(features_df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Adiciona indicadores técnicos ao DataFrame."""
    # Médias Móveis
    features_df['SMA_20'] = close.rolling(window=20).mean()
    features_df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features_df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    features_df['MACD_12_26_9'] = ema_fast - ema_slow
    features_df['MACDs_12_26_9'] = features_df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    features_df['MACDh_12_26_9'] = features_df['MACD_12_26_9'] - features_df['MACDs_12_26_9']
    
    # Bandas de Bollinger
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    features_df['BBM_20_2.0'] = sma_20
    features_df['BBU_20_2.0'] = sma_20 + (std_20 * 2)
    features_df['BBL_20_2.0'] = sma_20 - (std_20 * 2)
    
    return features_df


def _add_signal_features(features_df: pd.DataFrame, close: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de processamento de sinal (FFT e Wavelet)."""
    close_prices = close.values
    
    if len(close_prices) == 0:
        raise ValueError("Série de preços vazia. Não é possível calcular FFT.")
    
    # FFT Smoothing
    fft_vals = rfft(close_prices)
    if len(fft_vals) > 10:
        fft_vals[int(len(fft_vals) * 0.1):] = 0
    smoothed_fft = irfft(fft_vals)
    features_df['fft_smoothed'] = np.resize(smoothed_fft, len(df))
    
    # Wavelet Transform
    try:
        coeffs = pywt.wavedec(close_prices, 'db4', level=4)
        coeffs_denoised = coeffs[:]
        for i in range(1, len(coeffs_denoised)):
            coeffs_denoised[i] = pywt.threshold(
                coeffs_denoised[i], 
                value=np.std(coeffs_denoised[i]) / 2, 
                mode='soft'
            )
        denoised_prices = pywt.waverec(coeffs_denoised, 'db4')
        features_df['wavelet_denoised'] = np.resize(denoised_prices, len(df))
        
        for i, c in enumerate(coeffs):
            c_resized = np.resize(c, len(df))
            features_df[f'wavelet_coeff_{i}'] = pd.Series(c_resized, index=df.index)
    except Exception as e:
        print(f"Erro ao aplicar Wavelet: {e}")
    
    return features_df


def create_sequences(X: pd.DataFrame, y: pd.Series, time_steps: int = 20):
    """
    Cria sequências temporais para modelos de deep learning.
    
    Args:
        X: DataFrame de features
        y: Series de labels
        time_steps: Número de passos temporais por sequência
    
    Returns:
        Tuple (X_sequences, y_sequences, indices)
    """
    Xs, ys, idxs = [], [], []
    
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        label = y.iloc[i + time_steps]
        ys.append(label)
        idx = y.index[i + time_steps]
        idxs.append(idx)
    
    return np.array(Xs), np.array(ys), np.array(idxs)
