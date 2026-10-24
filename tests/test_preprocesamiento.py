"""Pruebas del preprocesamiento: forma, tipo, rango, inversión y errores."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import config
from src.imagenes.preprocesamiento import preprocesar


def _digito_sobre_fondo(fondo: int, trazo: int, alto: int = 40, ancho: int = 20) -> np.ndarray:
    imagen = np.full((80, 80), fondo, dtype=np.uint8)
    y0, x0 = (80 - alto) // 2, (80 - ancho) // 2
    cv2.rectangle(imagen, (x0, y0), (x0 + ancho, y0 + alto), trazo, thickness=-1)
    return imagen


def test_salida_28x28_float_normalizada() -> None:
    salida = preprocesar(_digito_sobre_fondo(fondo=255, trazo=0))

    assert salida.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert salida.dtype == np.float32
    assert 0.0 <= float(salida.min()) and float(salida.max()) <= 1.0
    assert float(salida.max()) > 0.5  # el trazo quedó claro sobre fondo oscuro


def test_invierte_fondo_claro_y_respeta_fondo_oscuro() -> None:
    clara = preprocesar(_digito_sobre_fondo(fondo=255, trazo=0))
    oscura = preprocesar(_digito_sobre_fondo(fondo=0, trazo=255))

    np.testing.assert_allclose(clara, oscura, atol=1e-6)


def test_conserva_proporcion_del_trazo() -> None:
    salida = preprocesar(_digito_sobre_fondo(fondo=255, trazo=0, alto=60, ancho=15))
    ys, xs = np.nonzero(salida > 0.5)

    alto_trazo = ys.max() - ys.min() + 1
    ancho_trazo = xs.max() - xs.min() + 1
    assert alto_trazo <= config.DIGIT_BOX
    assert 18 <= alto_trazo <= config.DIGIT_BOX
    assert ancho_trazo < alto_trazo


def test_imagen_sin_trazo_falla() -> None:
    with pytest.raises(ValueError, match="trazo"):
        preprocesar(np.full((64, 64), 255, dtype=np.uint8))
