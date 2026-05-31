"""
Loader para o dataset UWaveGestureLibrary do UEA Time Series Classification Archive.

Dataset MULTIVARIADO: 3 canais de acelerômetro (x, y, z), comprimento 315, 8 gestos.
Faz download (via aeon, com fallback de download direto do ZIP), converte labels
(strings "1".."8" → inteiros 0..7) e cria split estratificado de validação.

IMPORTANTE: ao contrário do FordA (univariado), aqui mantemos a dimensão de canais.
aeon retorna (n, n_channels, length); transpomos para (n, length, n_channels) —
a convenção (batch, time, features) esperada por base_models.build_model.
"""
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
import json
import urllib.request
import zipfile
import tempfile

_DATASET = "UWaveGestureLibrary"


class UWaveDataLoader:
    """
    Carrega e prepara o dataset UWaveGestureLibrary para classificação
    multivariada multiclasse (8 gestos, 3 canais).
    """

    # URL direta para download (UEA Archive via timeseriesclassification.com)
    _BASE_URL = (
        "https://www.timeseriesclassification.com/aeon-toolkit/UWaveGestureLibrary.zip"
    )

    def __init__(self, data_dir: Path, random_seed: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download(self, force: bool = False) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
        """
        Baixa UWaveGestureLibrary e retorna (X_train, y_train, X_test, y_test).

        X tem shape (n, length, n_channels) = (n, 315, 3).
        y são labels originais como strings ("1".."8").
        Tenta primeiro ``aeon``, depois download direto do ZIP.
        """
        cache_train = self.data_dir / "_raw_X_train.npy"
        cache_test = self.data_dir / "_raw_X_test.npy"

        if not force and cache_train.exists() and cache_test.exists():
            X_train = np.load(cache_train)
            y_train = np.load(self.data_dir / "_raw_y_train.npy", allow_pickle=True)
            X_test = np.load(cache_test)
            y_test = np.load(self.data_dir / "_raw_y_test.npy", allow_pickle=True)
            print(f"✓ {_DATASET} carregado do cache "
                  f"({len(X_train)} train, {len(X_test)} test, shape {X_train.shape[1:]})")
            return X_train, y_train, X_test, y_test

        # --- Tentar via aeon ---
        try:
            from aeon.datasets import load_classification
            X_train, y_train = load_classification(_DATASET, split="train")
            X_test, y_test = load_classification(_DATASET, split="test")
            # aeon retorna (n, n_channels, length) — transpor para (n, length, n_channels)
            X_train = np.transpose(X_train, (0, 2, 1)).astype(np.float32)
            X_test = np.transpose(X_test, (0, 2, 1)).astype(np.float32)
            print(f"✓ {_DATASET} baixado via aeon "
                  f"({len(X_train)} train, {len(X_test)} test, shape {X_train.shape[1:]})")
        except Exception as exc:
            print(f"aeon indisponível ({exc}); tentando download direto…")
            X_train, y_train, X_test, y_test = self._download_direct()

        # cache
        np.save(self.data_dir / "_raw_X_train.npy", X_train)
        np.save(self.data_dir / "_raw_y_train.npy", y_train)
        np.save(self.data_dir / "_raw_X_test.npy", X_test)
        np.save(self.data_dir / "_raw_y_test.npy", y_test)

        return X_train, y_train, X_test, y_test

    def _download_direct(self) -> Tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
        """Fallback: download direto do ZIP + parse dos arquivos .ts multivariados."""
        print(f"Baixando {_DATASET} diretamente do UEA Archive…")
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / f"{_DATASET}.zip"
            urllib.request.urlretrieve(self._BASE_URL, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)

            tmp_path = Path(tmp)
            train_file = self._find_file(tmp_path, "TRAIN")
            test_file = self._find_file(tmp_path, "TEST")

            X_train, y_train = self._parse_ts_file(train_file)
            X_test, y_test = self._parse_ts_file(test_file)

        print(f"✓ {_DATASET} baixado ({len(X_train)} train, {len(X_test)} test)")
        return X_train, y_train, X_test, y_test

    @staticmethod
    def _find_file(base: Path, keyword: str) -> Path:
        """Encontra arquivo de treino ou teste dentro do diretório extraído."""
        for ext in ("*.ts", "*.arff", "*.txt"):
            for f in base.rglob(ext):
                if keyword.upper() in f.name.upper():
                    return f
        raise FileNotFoundError(f"Não encontrou arquivo '{keyword}' em {base}")

    @staticmethod
    def _parse_ts_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse de arquivo .ts multivariado do UEA/sktime.

        Formato após @data:  dim1_v1,dim1_v2,...:dim2_v1,...:...:label
        Cada dimensão (canal) é separada por ':' e o último campo é o label.
        Retorna X com shape (n, length, n_channels) e y como strings.
        """
        lines = filepath.read_text().strip().splitlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("@data"):
                data_start = i + 1
                break

        X_list, y_list = [], []
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue

            parts = line.split(":")
            label = parts[-1].strip()
            channels = []
            for dim in parts[:-1]:
                vals = [float(v) for v in dim.replace(",", " ").split() if v.strip()]
                channels.append(vals)

            # channels: (n_channels, length) → (length, n_channels)
            arr = np.array(channels, dtype=np.float32).T
            X_list.append(arr)
            y_list.append(label)

        X = np.stack(X_list, axis=0)              # (n, length, n_channels)
        y = np.array(y_list, dtype=object)
        return X, y

    # ------------------------------------------------------------------
    # Preparação
    # ------------------------------------------------------------------
    def prepare(
        self,
        val_fraction: float = 0.15,
        force_download: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Pipeline completo: download → map labels → split de validação → salvar.

        Returns:
            Dict com X_train, y_train, X_val, y_val, X_test, y_test e class_mapping.
        """
        X_train, y_train_raw, X_test, y_test_raw = self.download(force=force_download)

        # Mapear labels (strings/ints originais) → inteiros 0..n_classes-1.
        # np.unique ordena de forma estável e reprodutível.
        classes = np.unique(np.concatenate([y_train_raw, y_test_raw]))
        class_mapping = {str(c): i for i, c in enumerate(classes)}

        def _map(y):
            return np.array([class_mapping[str(v)] for v in y], dtype=np.int64)

        y_train = _map(y_train_raw)
        y_test = _map(y_test_raw)
        n_classes = len(classes)

        # Split estratificado para validação
        X_train, X_val, y_train, y_val = self._stratified_split(
            X_train, y_train, val_fraction
        )

        # Salvar
        for arr, name in [
            (X_train, "X_train"), (y_train, "y_train"),
            (X_val, "X_val"), (y_val, "y_val"),
            (X_test, "X_test"), (y_test, "y_test"),
        ]:
            np.save(self.data_dir / f"{name}.npy", arr)

        # Metadata
        self.metadata = {
            "dataset": _DATASET,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "sequence_length": int(X_train.shape[1]),
            "n_features": int(X_train.shape[2]),
            "n_classes": int(n_classes),
            "class_mapping": class_mapping,
            "class_distribution_train": {
                str(c): int((y_train == c).sum()) for c in range(n_classes)
            },
            "class_distribution_val": {
                str(c): int((y_val == c).sum()) for c in range(n_classes)
            },
            "class_distribution_test": {
                str(c): int((y_test == c).sum()) for c in range(n_classes)
            },
            "val_fraction": val_fraction,
            "random_seed": self.random_seed,
        }
        with open(self.data_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\n✓ Dados salvos em {self.data_dir}")
        print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
        print(f"  Classes ({n_classes}): {class_mapping}")
        print(f"  Distrib. train: {self.metadata['class_distribution_train']}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "class_mapping": class_mapping,
        }

    def _stratified_split(
        self, X: np.ndarray, y: np.ndarray, val_fraction: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split estratificado para manter proporção de classes."""
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_fraction,
            random_state=self.random_seed,
            stratify=y,
        )
        return X_train, X_val, y_train, y_val

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
    @staticmethod
    def load_prepared(data_dir: Path) -> Dict[str, np.ndarray]:
        """Carrega dados já preparados do disco."""
        data_dir = Path(data_dir)
        data = {}
        for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
            path = data_dir / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"{path} não encontrado. Execute prepare() primeiro.")
            data[name] = np.load(path)
        return data

    @staticmethod
    def load_metadata(data_dir: Path) -> Dict[str, Any]:
        """Carrega metadata."""
        meta_path = Path(data_dir) / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}
