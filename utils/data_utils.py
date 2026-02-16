"""
Módulo para download e carregamento de dados financeiros.
"""
import yfinance as yf
import pandas as pd


def download_data(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Baixa dados históricos do Yahoo Finance para um único ticker.
    
    Args:
        ticker: Símbolo do ativo (ex: 'VALE3.SA')
        start_date: Data inicial no formato 'YYYY-MM-DD'
        end_date: Data final no formato 'YYYY-MM-DD' (None = hoje)
    
    Returns:
        DataFrame com colunas: Open, High, Low, Close, Adj Close, Volume
    
    Raises:
        ValueError: Se não for possível baixar dados para o ticker
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"Não foi possível baixar dados para o ticker {ticker}.")

    # Ajusta o cabeçalho
    data.reset_index(inplace=True)
    data = data.rename(columns={
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume"
    })
    data.set_index("Date", inplace=True)

    return data


def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV.
    
    Args:
        filepath: Caminho para o arquivo CSV
    
    Returns:
        DataFrame com índice datetime
    """
    return pd.read_csv(filepath, index_col="Date", parse_dates=True)


def save_data_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Salva DataFrame em arquivo CSV.
    
    Args:
        df: DataFrame a ser salvo
        filepath: Caminho do arquivo de destino
    """
    df.to_csv(filepath)
