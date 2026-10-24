"""Pruebas de segmentación: cantidad de cajas, orden de lectura y unión de fragmentos."""

from __future__ import annotations

import cv2
import numpy as np

from src.imagenes.segmentacion import (
    ordenar_lectura,
    recortar_digitos,
    segmentar_digitos,
    unir_fragmentos,
)


def _formulario(posiciones: list[tuple[int, int]], alto: int = 40, ancho: int = 24) -> np.ndarray:
    imagen = np.full((220, 520), 255, dtype=np.uint8)
    for x, y in posiciones:
        cv2.rectangle(imagen, (x, y), (x + ancho, y + alto), 0, thickness=-1)
    return imagen


def test_encuentra_cada_digito_separado() -> None:
    posiciones = [(40, 60), (120, 60), (200, 60), (280, 60)]
    cajas = segmentar_digitos(_formulario(posiciones))

    assert len(cajas) == 4
    ordenadas = ordenar_lectura(cajas)
    assert [caja[0] for caja in ordenadas] == sorted(pos[0] for pos in posiciones)
    assert len(recortar_digitos(_formulario(posiciones), ordenadas)) == 4


def test_orden_de_lectura_por_lineas() -> None:
    posiciones = [(300, 20), (40, 20), (180, 120), (60, 120)]
    ordenadas = ordenar_lectura(segmentar_digitos(_formulario(posiciones)))

    assert [caja[1] for caja in ordenadas[:2]] == [ordenadas[0][1]] * 2
    assert ordenadas[0][0] < ordenadas[1][0]
    assert ordenadas[2][0] < ordenadas[3][0]
    assert ordenadas[0][1] < ordenadas[2][1]


def test_une_fragmentos_verticales_del_mismo_digito() -> None:
    cajas = [(10, 10, 20, 15), (10, 27, 20, 15), (60, 10, 20, 30)]

    unidas = unir_fragmentos(cajas)

    assert len(unidas) == 2
    assert (10, 10, 20, 32) in unidas
    assert (60, 10, 20, 30) in unidas


def test_formulario_vacio_no_falla() -> None:
    cajas = segmentar_digitos(np.full((100, 100), 255, dtype=np.uint8))

    assert cajas == []
    assert ordenar_lectura(cajas) == []
