"""Pruebas de la selección de datos: solo dígitos, legibles, con trazo y sin duplicados."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config
from src.datos.seleccion import seleccionar


def _imagen_con_trazo(ruta: Path, valor_fondo: int = 255, valor_trazo: int = 0, digito: int = 0) -> None:
    imagen = np.full((64, 64), valor_fondo, dtype=np.uint8)
    # Cada dígito se dibuja en una posición distinta para que el hash sea único.
    x0 = 8 + (digito % 5) * 6
    y0 = 8 + (digito // 5) * 6
    cv2.rectangle(imagen, (x0, y0), (x0 + 22, y0 + 40), valor_trazo, thickness=-1)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), imagen)


def _imagen_en_blanco(ruta: Path, valor: int = 255) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), np.full((64, 64), valor, dtype=np.uint8))


def construir_crudo(base: Path) -> Path:
    """Arma un ``by_class`` mínimo con casos válidos e inválidos."""
    crudo = base / "by_class"
    for codigo in config.DIGIT_CODES:
        _imagen_con_trazo(
            crudo / codigo / "hsf_0" / f"valida_{codigo}.png",
            digito=int(codigo, 16) - 0x30,
        )

    _imagen_en_blanco(crudo / "30" / "hsf_0" / "blanca.png")
    (crudo / "30" / "hsf_0" / "corrupta.png").write_bytes(b"esto no es un png")
    shutil.copy2(crudo / "30" / "hsf_0" / "valida_30.png", crudo / "30" / "hsf_0" / "duplicada.png")

    # Carpeta que no es un dígito: debe ignorarse por completo.
    _imagen_con_trazo(crudo / "41" / "hsf_0" / "letra.png", digito=9)

    # Un .mit de ejemplo: debe revisarse, no seleccionarse.
    (crudo / "30" / "hsf_0.mit").write_text(
        "1\nvalida_30.png f0000_14/d0000_14/d0000_14_00000.png\n", encoding="utf-8"
    )
    return crudo


def test_seleccion_filtra_y_registra(tmp_path: Path) -> None:
    crudo = construir_crudo(tmp_path)
    manifiesto_path = tmp_path / "manifest.csv"

    manifiesto = seleccionar(raw_dir=crudo, manifest_path=manifiesto_path)

    assert manifiesto_path.is_file()
    assert list(pd.read_csv(manifiesto_path).columns) == config.MANIFEST_COLUMNS
    assert set(manifiesto["digito"].unique()) == set(range(10))
    assert not manifiesto["ruta_original"].str.contains("/41/").any()

    motivos = manifiesto.loc[~manifiesto["valida"], "motivo_descarte"].value_counts().to_dict()
    assert motivos.get("corrupta") == 1
    assert motivos.get("en_blanco") == 1
    assert motivos.get("duplicada") == 1
    assert int(manifiesto["valida"].sum()) == 10

    # El .mit se usa para trazabilidad: la imagen válida conoce su escritor.
    fila = manifiesto[manifiesto["hash"] == manifiesto.loc[manifiesto["valida"], "hash"].iloc[0]]
    assert fila["escritor"].iloc[0] in {"f0000", ""}
    assert set(manifiesto["particion"]) >= {"hsf_0"}
