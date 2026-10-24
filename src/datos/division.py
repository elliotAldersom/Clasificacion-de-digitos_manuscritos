"""División estratificada 80/20 de las imágenes válidas en ``data/fit`` y ``data/test``."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import asegurar_directorio, guardar_json, obtener_logger

logger = obtener_logger("division")


def repartir(
    manifiesto: pd.DataFrame,
    semilla: int = config.SEED,
    fit_ratio: float = config.FIT_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reparte las filas en fit/test conservando la proporción de cada dígito."""
    validas = manifiesto[manifiesto["valida"]].copy()
    if validas.empty:
        raise ValueError("El manifiesto no tiene imágenes válidas para dividir")
    if validas["hash"].duplicated().any():
        raise ValueError(
            "El manifiesto tiene hashes repetidos; ejecutá la etapa 'seleccion' antes de dividir"
        )

    indices_fit: list[int] = []
    indices_test: list[int] = []
    for digito in sorted(validas["digito"].unique()):
        grupo = validas[validas["digito"] == digito].sort_values("ruta_original")
        fit, test = train_test_split(
            grupo.index.to_numpy(),
            test_size=1.0 - fit_ratio,
            random_state=semilla,
            shuffle=True,
        )
        indices_fit.extend(fit.tolist())
        indices_test.extend(test.tolist())

    conjunto_fit = validas.loc[sorted(indices_fit)].copy()
    conjunto_test = validas.loc[sorted(indices_test)].copy()

    if not set(conjunto_fit["hash"]).isdisjoint(set(conjunto_test["hash"])):
        raise AssertionError("Fuga de datos: hay imágenes en fit y en test a la vez")
    return conjunto_fit, conjunto_test


def _copiar(conjunto: pd.DataFrame, destino: Path) -> int:
    copiadas = 0
    for fila in conjunto.itertuples():
        carpeta = asegurar_directorio(destino / str(fila.digito))
        shutil.copy2(fila.ruta_original, carpeta / f"{fila.hash}.png")
        copiadas += 1
    return copiadas


def dividir(
    manifest_path: Path = config.MANIFEST_PATH,
    fit_dir: Path = config.FIT_DIR,
    test_dir: Path = config.TEST_DIR,
    metrics_dir: Path | None = config.METRICS_DIR,
    semilla: int = config.SEED,
    fit_ratio: float = config.FIT_RATIO,
) -> dict[str, Any]:
    """Limpia, reparte y copia las imágenes; devuelve el resumen de la división."""
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(f"No existe el manifiesto: {manifest_path}")
    manifiesto = pd.read_csv(manifest_path)

    conjunto_fit, conjunto_test = repartir(manifiesto, semilla=semilla, fit_ratio=fit_ratio)

    for carpeta in (fit_dir, test_dir):
        if Path(carpeta).exists():
            shutil.rmtree(carpeta)
        asegurar_directorio(carpeta)

    logger.info("Copiando %d imágenes a fit y %d a test", len(conjunto_fit), len(conjunto_test))
    _copiar(conjunto_fit, Path(fit_dir))
    _copiar(conjunto_test, Path(test_dir))

    total = len(conjunto_fit) + len(conjunto_test)
    proporcion_test = len(conjunto_test) / total if total else 0.0
    por_digito = {}
    for digito in range(10):
        n_fit = int((conjunto_fit["digito"] == digito).sum())
        n_test = int((conjunto_test["digito"] == digito).sum())
        subtotal = n_fit + n_test
        por_digito[str(digito)] = {
            "fit": n_fit,
            "test": n_test,
            "total": subtotal,
            "proporcion_test": round(n_test / subtotal, 4) if subtotal else None,
        }

    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "semilla": semilla,
        "fit_ratio": fit_ratio,
        "total_fit": len(conjunto_fit),
        "total_test": len(conjunto_test),
        "total": total,
        "proporcion_test": round(proporcion_test, 4),
        "disjuntos": True,
        "por_digito": por_digito,
        "fit_dir": str(fit_dir),
        "test_dir": str(test_dir),
    }
    if metrics_dir is not None:
        guardar_json(Path(metrics_dir) / "division.json", resumen)
    logger.info(
        "División lista: fit=%d test=%d (%.2f %% en test)",
        len(conjunto_fit),
        len(conjunto_test),
        100 * proporcion_test,
    )
    return resumen
