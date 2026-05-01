"""
Módulo para validação cruzada com Purged K-Fold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedKFold(KFold):
    """
    K-Fold com Purga e Embargo para dados de séries temporais financeiras.

    Aceita tanto DataFrames com índice datetime quanto numpy arrays
    (nesse caso usa índice inteiro, e t1 deve ser uma pd.Series com
    índice inteiro ou valores inteiros/timestamps comparáveis).

    A purga remove do conjunto de treino as observações cujos labels
    se sobrepõem ao período de teste, evitando data leakage.

    O embargo adiciona um período de separação entre treino e teste
    para evitar que informações do futuro vazem para o modelo.

    Attributes:
        n_splits    : Número de dobras
        t1          : Series com event-end date/index para cada observação
        pct_embargo : Percentual de embargo (fração do dataset)
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series = None, pct_embargo: float = 0.01):
        super().__init__(n_splits=n_splits, shuffle=False)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Gera índices de treino e teste para cada dobra.

        Funciona com DataFrames (índice temporal) e com numpy arrays
        (usa RangeIndex inteiro; t1 deve ter os mesmos inteiros como índice).
        """
        if self.t1 is None:
            raise ValueError("t1 series must be provided.")

        n = len(X)

        # Suporte a numpy arrays: usar RangeIndex
        if hasattr(X, "index"):
            sample_index = X.index
        else:
            sample_index = pd.RangeIndex(n)

        # Alinhar t1 ao índice das amostras
        if len(self.t1) == n and not hasattr(self.t1, "index"):
            t1_aligned = pd.Series(self.t1.values, index=sample_index)
        else:
            t1_aligned = self.t1.reindex(sample_index)
            # Fallback: alinhar por posição se o reindex vazar NaN demais
            if t1_aligned.isna().mean() > 0.5:
                t1_aligned = pd.Series(self.t1.iloc[:n].values, index=sample_index)

        indices = np.arange(n)
        embargo_size = int(n * self.pct_embargo)
        test_splits = np.array_split(indices, self.n_splits)

        for i in range(self.n_splits):
            test_indices = test_splits[i]

            if len(test_indices) == 0:
                continue

            test_start = sample_index[test_indices[0]]
            test_end   = sample_index[test_indices[-1]]

            train_indices_all = np.concatenate([
                split for j, split in enumerate(test_splits) if i != j
            ])

            # Purga: remove amostras de treino cujo evento termina dentro do teste
            t1_train = t1_aligned.iloc[train_indices_all]
            purged_mask    = t1_train < test_start
            after_test_mask = sample_index[train_indices_all] > test_end
            keep_mask = purged_mask | after_test_mask

            train_indices = train_indices_all[keep_mask.values]

            # Embargo: remove amostras logo após o fim do teste
            if embargo_size > 0 and len(test_indices) > 0:
                embargo_end_idx = min(test_indices[-1] + embargo_size, n - 1)
                embargo_end = sample_index[embargo_end_idx]

                train_times   = sample_index[train_indices]
                embargo_mask  = ~((train_times > test_end) & (train_times <= embargo_end))
                train_indices = train_indices[embargo_mask]

            yield train_indices, test_indices


def create_train_test_split(
    df: pd.DataFrame,
    oos_years: int = 2
) -> tuple:
    """
    Divide dados em In-Sample (treino/validação) e Out-of-Sample (teste).

    Args:
        df       : DataFrame com dados completos (index datetime)
        oos_years: Anos para o conjunto Out-of-Sample

    Returns:
        Tuple (df_in_sample, df_oos_test)
    """
    split_date = df.index.max() - pd.DateOffset(years=oos_years)

    df_in_sample = df[df.index <= split_date]
    df_oos_test  = df[df.index > split_date]

    return df_in_sample, df_oos_test
