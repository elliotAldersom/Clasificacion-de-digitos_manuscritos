"""Carga de imágenes a arrays de NumPy con caché en ``data/processed``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config
from src.imagenes.preprocesamiento import preprocesar_archivo
from src.utils import asegurar_directorio, obtener_logger

logger = obtener_logger("cargador")

CACHE_VERSION = 1


def rutas_etiquetadas(directorio: Path) -> tuple[list[Path], np.ndarray]:
    """Lista las imágenes de ``directorio/<digito>/*.png`` con su etiqueta."""
    directorio = Path(directorio)
    if not directorio.is_dir():
        raise FileNotFoundError(f"No existe la partición: {directorio}")
    rutas: list[Path] = []
    etiquetas: list[int] = []
    for carpeta in sorted(directorio.iterdir()):
        if not carpeta.is_dir() or not carpeta.name.isdigit():
            continue
        digito = int(carpeta.name)
        archivos = sorted(carpeta.glob("*.png"))
        rutas.extend(archivos)
        etiquetas.extend([digito] * len(archivos))
    if not rutas:
        raise ValueError(f"La partición {directorio} está vacía")
    return rutas, np.asarray(etiquetas, dtype="int64")


def _ruta_cache(directorio: Path, tam: int) -> Path:
    return config.PROCESSED_DIR / f"{directorio.name}_{tam}x{tam}.npz"


def cargar_directorio(
    directorio: Path,
    tam: int = config.IMG_SIZE,
    lado_digito: int = config.DIGIT_BOX,
    usar_cache: bool = True,
    forzar: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Carga (y preprocesa) una partición completa; reutiliza la caché ``.npz``."""
    directorio = Path(directorio)
    cache = _ruta_cache(directorio, tam)
    if usar_cache and cache.is_file() and not forzar:
        datos = np.load(cache)
        if int(datos["version"]) == CACHE_VERSION and int(datos["tam"]) == tam:
            logger.info("Caché reutilizada: %s (%d imágenes)", cache.name, len(datos["y"]))
            return datos["X"], datos["y"]
        logger.info("Caché obsoleta (%s); se regenera", cache.name)

    rutas, etiquetas = rutas_etiquetadas(directorio)
    logger.info("Preprocesando %d imágenes de %s", len(rutas), directorio)
    X = np.zeros((len(rutas), tam, tam), dtype="float32")
    y = np.zeros(len(rutas), dtype="int64")
    destinos = 0
    for ruta, etiqueta in tqdm(list(zip(rutas, etiquetas)), desc=f"cargando {directorio.name}", leave=False):
        try:
            X[destinos] = preprocesar_archivo(ruta, tam=tam, lado_digito=lado_digito)
            y[destinos] = etiqueta
            destinos += 1
        except ValueError:
            logger.warning("Imagen descartada al cargar (sin trazo o ilegible): %s", ruta)
    X, y = X[:destinos], y[:destinos]

    if usar_cache:
        asegurar_directorio(config.PROCESSED_DIR)
        np.savez_compressed(cache, X=X, y=y, version=CACHE_VERSION, tam=tam)
        logger.info("Caché guardada: %s", cache)
    return X, y


def cargar_fit(usar_cache: bool = True, forzar: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Carga la partición de entrenamiento."""
    return cargar_directorio(config.FIT_DIR, usar_cache=usar_cache, forzar=forzar)


def cargar_test(usar_cache: bool = True, forzar: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Carga la partición de prueba (solo para evaluación final)."""
    return cargar_directorio(config.TEST_DIR, usar_cache=usar_cache, forzar=forzar)


def submuestra_estratificada(
    X: np.ndarray,
    y: np.ndarray,
    cantidad: int | None,
    semilla: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Toma una submuestra conservando la proporción de cada dígito."""
    if cantidad is None or cantidad >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=cantidad, random_state=semilla, stratify=y
    )
    return X_sub, y_sub


def a_entrada_cnn(X: np.ndarray) -> np.ndarray:
    """Añade el canal: ``(n, 28, 28)`` → ``(n, 28, 28, 1)``."""
    return X[..., np.newaxis]


def resumen_particion(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Resumen de tamaños y distribución de una partición."""
    valores, cuentas = np.unique(y, return_counts=True)
    return {
        "imagenes": int(len(y)),
        "forma": list(X.shape),
        "por_digito": {str(int(v)): int(c) for v, c in zip(valores, cuentas)},
    }
