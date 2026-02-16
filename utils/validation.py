"""
Módulo para validação cruzada com Purged K-Fold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedKFold(KFold):
    """
    K-Fold com Purga e Embargo para dados de séries temporais financeiras.
    
    A purga remove do conjunto de treino as observações cujos labels
    se sobrepõem ao período de teste, evitando data leakage.
    
    O embargo adiciona um período de separação entre treino e teste
    para evitar que informações do futuro vazem para o modelo.
    
    Attributes:
        n_splits: Número de dobras
        t1: Series com timestamps de fim do evento para cada observação
        pct_embargo: Percentual de embargo (fração do dataset)
    """
    
    def __init__(self, n_splits: int = 5, t1: pd.Series = None, pct_embargo: float = 0.01):
        """
        Inicializa o PurgedKFold.
        
        Args:
            n_splits: Número de dobras para validação cruzada
            t1: Series com timestamps de fim do evento (barreira vertical)
            pct_embargo: Fração do dataset para período de embargo
        """
        super().__init__(n_splits=n_splits, shuffle=False)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Gera índices de treino e teste para cada dobra.
        
        Args:
            X: DataFrame ou array de features
            y: Labels (opcional)
            groups: Grupos (não usado)
        
        Yields:
            Tuple (train_indices, test_indices) para cada dobra
        
        Raises:
            ValueError: Se t1 não foi fornecido
        """
        if self.t1 is None:
            raise ValueError("t1 series must be provided.")
        
        t1_aligned = self.t1.reindex(X.index).dropna()
        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)
        test_splits = np.array_split(indices, self.n_splits)
        
        for i in range(self.n_splits):
            test_indices = test_splits[i]
            
            if len(test_indices) == 0:
                continue
            
            test_start_time = X.index[test_indices[0]]
            
            # Combina índices de treino de outras dobras
            train_indices_all = np.concatenate([
                split for j, split in enumerate(test_splits) if i != j
            ])
            
            # Aplica purga: remove observações cujo t1 >= test_start_time
            t1_train = t1_aligned.iloc[train_indices_all]
            purged_train_mask = t1_train < test_start_time
            train_indices = t1_train[purged_train_mask].index
            train_indices = X.index.get_indexer(train_indices)
            
            # Aplica embargo
            if len(test_indices) > 0:
                test_end_time = X.index[test_indices[-1]]
                embargo_start_time = test_end_time + pd.Timedelta(days=1)
                embargo_end_time = embargo_start_time + pd.Timedelta(days=embargo_size)
                
                embargo_mask = (
                    (X.index[train_indices] < embargo_start_time) | 
                    (X.index[train_indices] > embargo_end_time)
                )
                train_indices = train_indices[embargo_mask]
            
            yield train_indices, test_indices


def create_train_test_split(
    df: pd.DataFrame, 
    oos_years: int = 2
) -> tuple:
    """
    Divide dados em In-Sample (treino/validação) e Out-of-Sample (teste).
    
    Args:
        df: DataFrame com dados completos (index datetime)
        oos_years: Anos para o conjunto Out-of-Sample
    
    Returns:
        Tuple (df_in_sample, df_oos_test)
    """
    split_date = df.index.max() - pd.DateOffset(years=oos_years)
    
    df_in_sample = df[df.index <= split_date]
    df_oos_test = df[df.index > split_date]
    
    return df_in_sample, df_oos_test
