"""Pruebas de la división 80/20: proporción, estratificación, disjunción e idempotencia."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config
from src.datos.division import dividir, repartir


def _manifiesto_falso(base: Path, por_digito: int = 100) -> Path:
    """Crea archivos reales y un manifiesto con ``por_digito`` filas por clase."""
    filas = []
    for digito in range(10):
        carpeta = base / "puro" / str(digito)
        carpeta.mkdir(parents=True, exist_ok=True)
        for indice in range(por_digito):
            ruta = carpeta / f"{digito}_{indice}.png"
            imagen = np.full((8, 8), 255, dtype=np.uint8)
            imagen[2:6, 2:6] = 10 * digito + 1
            imagen[0, indice % 8] = indice
            cv2.imwrite(str(ruta), imagen)
            contenido = hashlib.md5(ruta.read_bytes()).hexdigest()
            filas.append(
                {
                    "ruta_original": str(ruta),
                    "digito": digito,
                    "particion": "hsf_0",
                    "escritor": f"f{indice:04d}",
                    "hash": contenido,
                    "valida": True,
                    "motivo_descarte": "",
                }
            )
    manifiesto = base / "manifest.csv"
    pd.DataFrame(filas, columns=config.MANIFEST_COLUMNS).to_csv(manifiesto, index=False)
    return manifiesto


def test_reparto_estratificado_y_disjunto(tmp_path: Path) -> None:
    manifiesto_path = _manifiesto_falso(tmp_path)
    manifiesto = pd.read_csv(manifiesto_path)

    fit, test = repartir(manifiesto, semilla=config.SEED)

    assert len(fit) == 800
    assert len(test) == 200
    assert set(fit["hash"]).isdisjoint(set(test["hash"]))
    for digito in range(10):
        n_fit = int((fit["digito"] == digito).sum())
        n_test = int((test["digito"] == digito).sum())
        assert n_fit == 80 and n_test == 20

    fit_otra_vez, test_otra_vez = repartir(manifiesto, semilla=config.SEED)
    assert fit["hash"].tolist() == fit_otra_vez["hash"].tolist()
    assert test["hash"].tolist() == test_otra_vez["hash"].tolist()


def test_dividir_copia_e_idempotente(tmp_path: Path) -> None:
    manifiesto_path = _manifiesto_falso(tmp_path)
    fit_dir = tmp_path / "fit"
    test_dir = tmp_path / "test"

    primero = dividir(manifiesto_path, fit_dir, test_dir, metrics_dir=None)
    segundo = dividir(manifiesto_path, fit_dir, test_dir, metrics_dir=None)

    assert primero["total_fit"] == segundo["total_fit"] == 800
    assert primero["total_test"] == segundo["total_test"] == 200
    assert abs(segundo["proporcion_test"] - 0.20) < 0.01
    assert len(list(fit_dir.rglob("*.png"))) == 800
    assert len(list(test_dir.rglob("*.png"))) == 200
    for digito in range(10):
        assert (fit_dir / str(digito)).is_dir()
        assert (test_dir / str(digito)).is_dir()
