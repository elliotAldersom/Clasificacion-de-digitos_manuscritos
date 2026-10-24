"""Utilidades sin lógica de dominio: logging, semillas, JSON y hashes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from src import config

LOGGER_NAME = "tecnoforms"


def configurar_logging(nivel: int = logging.INFO, ruta_archivo: Path | None = None) -> None:
    """Configura el logger del proyecto: consola siempre y archivo en ``logs/pipeline.log``.

    El directorio de destino se crea si no existe. Si el archivo no se puede abrir
    (por ejemplo, un sistema de archivos de solo lectura), se sigue registrando
    únicamente en consola para no interrumpir la etapa en curso.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(nivel)
        return

    formato = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    manejador_consola = logging.StreamHandler()
    manejador_consola.setFormatter(formato)
    logger.addHandler(manejador_consola)

    destino = ruta_archivo if ruta_archivo is not None else config.LOG_PATH
    try:
        asegurar_directorio(destino.parent)
        manejador_archivo = logging.FileHandler(destino, mode="a", encoding="utf-8")
    except OSError as error:
        logger.warning("No se pudo abrir el archivo de log %s: %s", destino, error)
    else:
        manejador_archivo.setFormatter(formato)
        logger.addHandler(manejador_archivo)

    logger.setLevel(nivel)
    logger.propagate = False


def obtener_logger(nombre: str | None = None) -> logging.Logger:
    """Devuelve el logger del proyecto (o un hijo con el nombre indicado)."""
    if nombre is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{nombre}")


def fijar_semilla(semilla: int, con_tensorflow: bool = False) -> None:
    """Fija la semilla de todos los generadores aleatorios del proceso."""
    random.seed(semilla)
    np.random.seed(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if con_tensorflow:
        import keras

        keras.utils.set_random_seed(semilla)


def asegurar_directorio(ruta: Path) -> Path:
    """Crea el directorio (y sus padres) si no existe y lo devuelve."""
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def guardar_json(ruta: Path, datos: Any) -> Path:
    """Guarda un objeto serializable en JSON con formato legible."""
    asegurar_directorio(ruta.parent)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
    return ruta


def cargar_json(ruta: Path) -> Any:
    """Carga un JSON desde disco."""
    with Path(ruta).open(encoding="utf-8") as archivo:
        return json.load(archivo)


def hash_archivo(ruta: Path, algoritmo: str = "md5") -> str:
    """Calcula el hash del contenido de un archivo en bloques."""
    resumen = hashlib.new(algoritmo)
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 16), b""):
            resumen.update(bloque)
    return resumen.hexdigest()
