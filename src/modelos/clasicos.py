"""Modelos clásicos de comparación: KNN, SVM (RBF) y regresión logística."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src import config
from src.utils import asegurar_directorio


def construir_modelos() -> dict[str, Pipeline]:
    """Devuelve un pipeline por modelo (preprocesamiento + clasificador juntos)."""
    return {
        "knn": Pipeline(
            [
                ("pca", PCA(n_components=50, random_state=config.SEED)),
                ("modelo", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "svm": Pipeline(
            [
                ("pca", PCA(n_components=50, random_state=config.SEED)),
                ("modelo", SVC(kernel="rbf", C=5, gamma="scale")),
            ]
        ),
        "regresion_logistica": Pipeline(
            [
                ("modelo", LogisticRegression(max_iter=1000)),
            ]
        ),
    }


def rejillas_hiperparametros(rapido: bool = False) -> dict[str, dict[str, list[Any]]]:
    """Rejillas pequeñas para ``GridSearchCV`` (se reducen en modo rápido)."""
    if rapido:
        return {
            "knn": {"modelo__n_neighbors": [3, 5]},
            "svm": {"modelo__C": [1, 5]},
            "regresion_logistica": {"modelo__C": [1.0]},
        }
    return {
        "knn": {"modelo__n_neighbors": [3, 5, 7]},
        "svm": {"modelo__C": [1, 5, 10], "modelo__gamma": ["scale", 0.01]},
        "regresion_logistica": {"modelo__C": [1.0, 10.0]},
    }


def guardar_modelo(modelo: Any, nombre: str, models_dir: Path = config.MODELS_DIR) -> Path:
    """Serializa un modelo entrenado con joblib."""
    asegurar_directorio(Path(models_dir))
    ruta = Path(models_dir) / f"{nombre}.joblib"
    joblib.dump(modelo, ruta)
    return ruta


def cargar_modelo(nombre: str, models_dir: Path = config.MODELS_DIR) -> Any:
    """Carga un modelo serializado con joblib."""
    ruta = Path(models_dir) / f"{nombre}.joblib"
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el modelo {ruta}; ejecutá la etapa de entrenamiento")
    return joblib.load(ruta)
