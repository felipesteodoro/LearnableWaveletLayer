"""
Loader para o dataset FordA do UCR Time Series Classification Archive.

Faz download, converte labels (-1/1 → 0/1) e cria split de validação.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path
import json
import urllib.request
import zipfile
import tempfile
import shutil


class FordADataLoader:
    """
    Carrega e prepara o dataset FordA para classificação binária.
    """

    # URL direta para download (UCR/UEA Archive via timeseriesclassification.com)
    _BASE_URL = (
        "https://www.timeseriesclassification.com/aeon-toolkit/FordA.zip"
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
        Baixa FordA e retorna (X_train, y_train, X_test, y_test) com labels
        originais (-1 / 1).  Tenta primeiro ``aeon``, depois download direto.
        """
        cache_train = self.data_dir / "_raw_X_train.npy"
        cache_test = self.data_dir / "_raw_X_test.npy"

        if not force and cache_train.exists() and cache_test.exists():
            X_train = np.load(cache_train)
            y_train = np.load(self.data_dir / "_raw_y_train.npy")
            X_test = np.load(cache_test)
            y_test = np.load(self.data_dir / "_raw_y_test.npy")
            print(f"✓ FordA carregado do cache ({len(X_train)} train, {len(X_test)} test)")
            return X_train, y_train, X_test, y_test

        # --- Tentar via aeon ---
        try:
            from aeon.datasets import load_classification
            X_train, y_train = load_classification("FordA", split="train")
            X_test, y_test = load_classification("FordA", split="test")
            # aeon retorna (n, 1, L) — squeeze para (n, L)
            X_train = np.squeeze(X_train, axis=1)
            X_test = np.squeeze(X_test, axis=1)
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            print(f"✓ FordA baixado via aeon ({len(X_train)} train, {len(X_test)} test)")
        except Exception:
            X_train, y_train, X_test, y_test = self._download_direct()

        # cache
        for arr, name in [(X_train, "_raw_X_train"), (y_train, "_raw_y_train"),
                          (X_test, "_raw_X_test"), (y_test, "_raw_y_test")]:
            np.save(self.data_dir / f"{name}.npy", arr)

        return X_train, y_train, X_test, y_test

    def _download_direct(self) -> Tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
        """Fallback: download direto do ZIP + parse dos .ts/.tsv files."""
        print("Baixando FordA diretamente do UCR Archive…")
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "FordA.zip"
            urllib.request.urlretrieve(self._BASE_URL, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)

            # Procura arquivos .ts ou .tsv
            tmp_path = Path(tmp)
            train_file = self._find_file(tmp_path, "TRAIN")
            test_file = self._find_file(tmp_path, "TEST")

            X_train, y_train = self._parse_ts_file(train_file)
            X_test, y_test = self._parse_ts_file(test_file)

        print(f"✓ FordA baixado ({len(X_train)} train, {len(X_test)} test)")
        return X_train, y_train, X_test, y_test

    @staticmethod
    def _find_file(base: Path, keyword: str) -> Path:
        """Encontra arquivo de treino ou teste dentro do diretório extraído."""
        for ext in ("*.ts", "*.tsv", "*.txt", "*.arff"):
            for f in base.rglob(ext):
                if keyword.upper() in f.name.upper():
                    return f
        raise FileNotFoundError(
            f"Não encontrou arquivo '{keyword}' em {base}"
        )

    @staticmethod
    def _parse_ts_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse genérico de arquivos .ts do UCR / aeon.
        Suporta formato .ts (com @data section) e formato TSV simples.
        """
        lines = filepath.read_text().strip().splitlines()

        # Detecta formato @data
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("@data"):
                data_start = i + 1
                break
            # Se a primeira coluna é numérica, trata como TSV direto
            parts = line.strip().split()
            if len(parts) > 10:
                data_start = 0
                break

        X_list, y_list = [], []
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue

            # Tenta `:` como separador (formato .ts do aeon)
            if ":" in line:
                # formato: dim1_vals : label  OU  label : dim1_vals
                # O padrão aeon/sktime é: class : dim_values
                # Mas FordA.ts usa: dim_values (comma-sep) : class no final
                # Precisamos detectar
                parts = line.split(":")
                # Última parte não-vazia é geralmente o label
                label_str = parts[-1].strip()
                # Se o label parece numérico simples
                try:
                    label = float(label_str)
                    # valores são as partes anteriores
                    vals_str = ":".join(parts[:-1])
                    vals = [float(v) for v in vals_str.replace(",", " ").split() if v.strip()]
                except ValueError:
                    # label no início
                    label = float(parts[0].strip())
                    vals_str = ":".join(parts[1:])
                    vals = [float(v) for v in vals_str.replace(",", " ").split() if v.strip()]
            else:
                # Formato TSV ou espaço-separado (label na primeira coluna)
                vals = line.replace(",", " ").split()
                label = float(vals[0])
                vals = [float(v) for v in vals[1:]]

            y_list.append(label)
            X_list.append(vals)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
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
        Pipeline completo: download → map labels → val split → salvar.

        Returns:
            Dict com X_train, y_train, X_val, y_val, X_test, y_test
        """
        X_train, y_train, X_test, y_test = self.download(force=force_download)

        # Mapear labels: -1 → 0,  1 → 1
        y_train = np.where(y_train == -1, 0, 1).astype(np.int64)
        y_test = np.where(y_test == -1, 0, 1).astype(np.int64)

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
            "dataset": "FordA",
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "sequence_length": int(X_train.shape[1]),
            "n_classes": 2,
            "class_distribution_train": {
                str(c): int((y_train == c).sum()) for c in [0, 1]
            },
            "class_distribution_val": {
                str(c): int((y_val == c).sum()) for c in [0, 1]
            },
            "class_distribution_test": {
                str(c): int((y_test == c).sum()) for c in [0, 1]
            },
            "val_fraction": val_fraction,
            "random_seed": self.random_seed,
        }
        with open(self.data_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\n✓ Dados salvos em {self.data_dir}")
        print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
        print(f"  Classes train: {self.metadata['class_distribution_train']}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
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
