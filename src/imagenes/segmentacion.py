"""Segmentación de dígitos dentro de un formulario mediante componentes conectados."""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from src import config

Caja = tuple[int, int, int, int]


def binarizar(imagen: np.ndarray) -> np.ndarray:
    """Devuelve una máscara con el trazo en blanco (asume fondo claro)."""
    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)
    _, binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binaria


def segmentar_digitos(
    imagen_gris: np.ndarray,
    area_min: int = config.AREA_MINIMA,
    area_max_ratio: float = 0.9,
) -> list[Caja]:
    """Devuelve las cajas ``(x, y, ancho, alto)`` de cada trazo encontrado."""
    binaria = binarizar(imagen_gris)
    total_pixeles = binaria.size
    n_etiquetas, _, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)

    cajas: list[Caja] = []
    for i in range(1, n_etiquetas):  # 0 es el fondo
        x, y, w, h, area = (int(v) for v in stats[i, :5])
        if area < area_min:
            continue
        if area > area_max_ratio * total_pixeles:  # marco o fondo residual
            continue
        cajas.append((x, y, w, h))
    return cajas


def unir_fragmentos(
    cajas: Sequence[Caja],
    solape_horizontal: float = 0.6,
    brecha_vertical: int = 8,
) -> list[Caja]:
    """Une cajas que son partes de un mismo dígito (trazo partido).

    Dos cajas se unen solo si comparten buena parte de su ancho (arriba/abajo),
    nunca si están una al lado de la otra.
    """
    restantes = list(cajas)
    unidas = True
    while unidas:
        unidas = False
        for i in range(len(restantes)):
            for j in range(i + 1, len(restantes)):
                ax, ay, aw, ah = restantes[i]
                bx, by, bw, bh = restantes[j]
                solape = min(ax + aw, bx + bw) - max(ax, bx)
                if solape <= 0:
                    continue
                proporcion = solape / max(1, min(aw, bw))
                brecha = max(ay, by) - min(ay + ah, by + bh)
                if proporcion >= solape_horizontal and brecha <= brecha_vertical:
                    x0, y0 = min(ax, bx), min(ay, by)
                    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    restantes[i] = (x0, y0, x1 - x0, y1 - y0)
                    restantes.pop(j)
                    unidas = True
                    break
            if unidas:
                break
    return restantes


def ordenar_lectura(cajas: Iterable[Caja], tolerancia_linea: int | None = None) -> list[Caja]:
    """Ordena las cajas por línea (centro vertical) y de izquierda a derecha."""
    cajas = sorted(cajas, key=lambda caja: caja[1] + caja[3] / 2)
    if not cajas:
        return []
    if tolerancia_linea is None:
        alturas = [caja[3] for caja in cajas]
        tolerancia_linea = max(config.TOLERANCIA_LINEA, int(np.median(alturas) * 0.5))

    lineas: list[list[Caja]] = []
    actual = [cajas[0]]
    for caja in cajas[1:]:
        centro = caja[1] + caja[3] / 2
        centro_linea = sum(c[1] + c[3] / 2 for c in actual) / len(actual)
        if abs(centro - centro_linea) <= tolerancia_linea:
            actual.append(caja)
        else:
            lineas.append(sorted(actual, key=lambda c: c[0]))
            actual = [caja]
    lineas.append(sorted(actual, key=lambda c: c[0]))
    return [caja for linea in lineas for caja in linea]


def recortar_digitos(imagen: np.ndarray, cajas: Sequence[Caja]) -> list[np.ndarray]:
    """Extrae cada caja como un recorte independiente."""
    alto, ancho = imagen.shape[:2]
    recortes = []
    for x, y, w, h in cajas:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(ancho, x + w), min(alto, y + h)
        if x1 > x0 and y1 > y0:
            recortes.append(imagen[y0:y1, x0:x1])
    return recortes


def segmentar_formulario(imagen: np.ndarray) -> tuple[list[Caja], list[np.ndarray]]:
    """Segmenta, une fragmentos y ordena en una sola llamada."""
    cajas = unir_fragmentos(segmentar_digitos(imagen))
    cajas = ordenar_lectura(cajas)
    return cajas, recortar_digitos(imagen, cajas)
