"""Selección de los datos útiles del crudo: solo dígitos, legibles, con trazo y sin duplicados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.datos.inspeccion import mapa_escritores
from src.utils import asegurar_directorio, hash_archivo, obtener_logger

logger = obtener_logger("seleccion")


def motivo_invalidez(ruta: Path) -> tuple[bool, str]:
    """Comprueba que la imagen sea legible y contenga trazo."""
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return False, "corrupta"
    if float(imagen.std()) < 1.0:
        return False, "en_blanco"
    return True, ""


def seleccionar(
    raw_dir: Path = config.RAW_DIR,
    manifest_path: Path = config.MANIFEST_PATH,
) -> pd.DataFrame:
    """Recorre las carpetas de dígitos y escribe ``data/manifest.csv``.

    Cada archivo recibe una fila con su dígito, partición de origen, escritor
    (si los ``.mit`` lo exponen), hash, validez y motivo de descarte.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de datos crudos: {raw_dir}")

    vistos: dict[str, str] = {}
    filas: list[dict[str, Any]] = []

    for codigo in config.DIGIT_CODES:
        carpeta = raw_dir / codigo
        digito = int(codigo, 16) - 0x30
        if not carpeta.is_dir():
            logger.warning("Falta la carpeta del dígito %s (%s); se omite", digito, codigo)
            continue
        escritores = mapa_escritores(carpeta)
        archivos = sorted(carpeta.rglob("*.png"))
        logger.info("Dígito %d: %d imágenes candidatas", digito, len(archivos))

        for ruta in tqdm(archivos, desc=f"dígito {digito}", leave=False):
            resumen = hash_archivo(ruta)
            valida = True
            motivo = ""
            if resumen in vistos:
                valida, motivo = False, "duplicada"
            else:
                vistos[resumen] = str(ruta)
                valida, motivo = motivo_invalidez(ruta)
            filas.append(
                {
                    "ruta_original": str(ruta),
                    "digito": digito,
                    "particion": ruta.parent.name,
                    "escritor": escritores.get(ruta.name, ""),
                    "hash": resumen,
                    "valida": valida,
                    "motivo_descarte": motivo,
                }
            )

    manifiesto = pd.DataFrame(filas, columns=config.MANIFEST_COLUMNS)
    if manifiesto.empty:
        raise ValueError(f"No se encontró ninguna imagen en {raw_dir}")

    asegurar_directorio(manifest_path.parent)
    manifiesto.to_csv(manifest_path, index=False)

    validas = manifiesto[manifiesto["valida"]]
    descartes = manifiesto[~manifiesto["valida"]]["motivo_descarte"].value_counts().to_dict()
    logger.info(
        "Manifiesto: %d imágenes, %d válidas, descartes=%s → %s",
        len(manifiesto),
        len(validas),
        descartes or "ninguno",
        manifest_path,
    )
    return manifiesto
