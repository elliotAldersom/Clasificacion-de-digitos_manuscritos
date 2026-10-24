"""Preprocesamiento de imágenes: grises, recorte, escalado y lienzo 28x28."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src import config


def a_escala_grises(imagen: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR/BGRA a un único canal en escala de grises."""
    if imagen.ndim == 2:
        return imagen
    if imagen.ndim == 3 and imagen.shape[2] == 1:
        return imagen[:, :, 0]
    if imagen.ndim == 3 and imagen.shape[2] == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if imagen.ndim == 3 and imagen.shape[2] == 4:
        return cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Formato de imagen no soportado: shape={imagen.shape}")


def normalizar_a_uint8(imagen: np.ndarray) -> np.ndarray:
    """Garantiza una imagen de 8 bits sin recortar información útil."""
    if imagen.dtype == np.uint8:
        return imagen
    recortada = np.clip(imagen, 0, 255)
    return recortada.astype(np.uint8)


def invertir_si_fondo_claro(imagen: np.ndarray) -> np.ndarray:
    """Deja el trazo claro sobre fondo oscuro (estilo MNIST)."""
    if float(np.median(imagen)) > 127:
        return cv2.bitwise_not(imagen)
    return imagen


def preprocesar(
    imagen: np.ndarray,
    tam: int = config.IMG_SIZE,
    lado_digito: int = config.DIGIT_BOX,
) -> np.ndarray:
    """Convierte una imagen de un dígito en un lienzo ``tam×tam`` normalizado.

    Pasos: escala de grises, fondo oscuro, binarización de apoyo (Otsu), recorte
    al trazo, escalado conservando la proporción y centrado en el lienzo.

    Raises:
        ValueError: si la imagen no contiene ningún trazo.
    """
    gris = normalizar_a_uint8(a_escala_grises(imagen))
    gris = invertir_si_fondo_claro(gris)

    _, mascara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mascara)
    if ys.size == 0:
        raise ValueError("La imagen no tiene trazo")

    recorte = gris[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    escala = lado_digito / max(recorte.shape)
    nuevo_tam = (
        max(1, round(recorte.shape[1] * escala)),
        max(1, round(recorte.shape[0] * escala)),
    )
    interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    recorte = cv2.resize(recorte, nuevo_tam, interpolation=interpolacion)

    lienzo = np.zeros((tam, tam), dtype=np.uint8)
    y0 = (tam - recorte.shape[0]) // 2
    x0 = (tam - recorte.shape[1]) // 2
    lienzo[y0 : y0 + recorte.shape[0], x0 : x0 + recorte.shape[1]] = recorte
    return lienzo.astype("float32") / 255.0


def leer_imagen_gris(ruta: Path) -> np.ndarray:
    """Lee una imagen en escala de grises y falla con un mensaje claro."""
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    return imagen


def preprocesar_archivo(ruta: Path, tam: int = config.IMG_SIZE, lado_digito: int = config.DIGIT_BOX) -> np.ndarray:
    """Atajo: lee un archivo y lo preprocesa con la misma función del entrenamiento."""
    return preprocesar(leer_imagen_gris(ruta), tam=tam, lado_digito=lado_digito)
