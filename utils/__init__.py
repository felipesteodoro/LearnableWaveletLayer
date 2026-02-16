# Trading Strategy Utils
# Módulos para estratégia de trading com CNN-LSTM e Triple Barrier Labeling

from .data_utils import download_data, load_data_from_csv
from .features import create_features, create_sequences
from .labeling import get_daily_vol, triple_barrier_labeling
from .model import create_cnn_lstm_model, create_model_for_tuning
from .validation import PurgedKFold
from .evaluation import calculate_financial_metrics, plot_labels_on_price, plot_confusion_matrix

__all__ = [
    'download_data',
    'load_data_from_csv',
    'create_features',
    'create_sequences',
    'get_daily_vol',
    'triple_barrier_labeling',
    'create_cnn_lstm_model',
    'create_model_for_tuning',
    'PurgedKFold',
    'calculate_financial_metrics',
    'plot_labels_on_price',
    'plot_confusion_matrix',
]
