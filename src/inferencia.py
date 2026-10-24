"""Lectura de un formulario sintético de punta a punta: segmentar → preprocesar → predecir."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src import config
from src.imagenes.preprocesamiento import preprocesar
from src.imagenes.segmentacion import ordenar_lectura, recortar_digitos, segmentar_digitos, unir_fragmentos
from src.utils import asegurar_directorio, guardar_json, obtener_logger

logger = obtener_logger("inferencia")

ALTO_DIGITO = 48
SEPARACION_MIN = 18
SEPARACION_MAX = 34


def _indice_test(test_dir: Path) -> dict[int, list[Path]]:
    indice: dict[int, list[Path]] = {}
    for digito in range(10):
        archivos = sorted((Path(test_dir) / str(digito)).glob("*.png"))
        if not archivos:
            raise FileNotFoundError(f"No hay imágenes de prueba para el dígito {digito}")
        indice[digito] = archivos
    return indice


def _pegar_digito(lienzo: np.ndarray, imagen: np.ndarray, x: int, y: int) -> None:
    """Pega un dígito oscuro sobre fondo claro conservando el trazo más oscuro."""
    alto, ancho = imagen.shape[:2]
    alto_lienzo, ancho_lienzo = lienzo.shape[:2]
    x1, y1 = min(ancho_lienzo, x + ancho), min(alto_lienzo, y + alto)
    if x1 <= x or y1 <= y:
        return
    region = lienzo[y:y1, x:x1]
    np.minimum(region, imagen[: y1 - y, : x1 - x], out=region)


def generar_formulario(
    indice: dict[int, list[Path]],
    rng: random.Random,
    digitos_por_campo: int = config.DIGITOS_POR_CAMPO,
    campos: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Genera un formulario sintético con ``campos`` filas de dígitos espaciados."""
    ancho = digitos_por_campo * (ALTO_DIGITO + SEPARACION_MAX) + 80
    alto = campos * (ALTO_DIGITO + 50) + 40
    lienzo = np.full((alto, ancho), 255, dtype=np.uint8)
    esperados: list[str] = []

    for campo in range(campos):
        y_base = 30 + campo * (ALTO_DIGITO + 50) + rng.randint(-4, 4)
        x = 40
        texto = ""
        for _ in range(digitos_por_campo):
            digito = rng.randint(0, 9)
            imagen = cv2.imread(str(rng.choice(indice[digito])), cv2.IMREAD_GRAYSCALE)
            if imagen is None:
                raise ValueError("No se pudo leer la imagen de prueba")
            escala = ALTO_DIGITO / imagen.shape[0]
            redimensionada = cv2.resize(
                imagen,
                (max(1, round(imagen.shape[1] * escala)), ALTO_DIGITO),
                interpolation=cv2.INTER_AREA,
            )
            _pegar_digito(lienzo, redimensionada, x, y_base)
            x += redimensionada.shape[1] + rng.randint(SEPARACION_MIN, SEPARACION_MAX)
            texto += str(digito)
        esperados.append(texto)

    ruido = rng.randint(0, 8)
    if ruido:
        ruido_gaussiano = np.random.default_rng(rng.randint(0, 2**31)).normal(0, ruido, lienzo.shape)
        lienzo = np.clip(lienzo + ruido_gaussiano, 0, 255).astype(np.uint8)
    return lienzo, esperados


def agrupar_por_lineas(cajas: Sequence[tuple[int, int, int, int]], tolerancia: int = 30) -> list[list[tuple[int, int, int, int]]]:
    """Agrupa cajas ordenadas en filas según su centro vertical."""
    if not cajas:
        return []
    lineas: list[list[tuple[int, int, int, int]]] = [[cajas[0]]]
    for caja in cajas[1:]:
        centro = caja[1] + caja[3] / 2
        centros_linea = [c[1] + c[3] / 2 for c in lineas[-1]]
        if abs(centro - sum(centros_linea) / len(centros_linea)) <= tolerancia:
            lineas[-1].append(caja)
        else:
            lineas.append([caja])
    return lineas


def leer_formulario(imagen_gris: np.ndarray, modelo: Any) -> list[str]:
    """Segmenta el formulario y devuelve un texto por línea detectada."""
    cajas = ordenar_lectura(unir_fragmentos(segmentar_digitos(imagen_gris)))
    recortes = recortar_digitos(imagen_gris, cajas)
    if not recortes:
        return []
    entradas = np.stack([preprocesar(recorte) for recorte in recortes])[..., None]
    predicciones = modelo.predict(entradas, verbose=0).argmax(axis=1)

    lineas: list[str] = []
    indice = 0
    for linea in agrupar_por_lineas(cajas):
        texto = "".join(str(int(predicciones[indice + i])) for i in range(len(linea)))
        lineas.append(texto)
        indice += len(linea)
    return lineas


def _anotar(imagen_gris: np.ndarray, esperados: Sequence[str], obtenidos: Sequence[str]) -> np.ndarray:
    """Dibuja las cajas y predicciones sobre el formulario."""
    color = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    cajas = ordenar_lectura(unir_fragmentos(segmentar_digitos(imagen_gris)))
    esperado = "".join(esperados)
    obtenido = "".join(obtenidos)
    for posicion, (x, y, w, h) in enumerate(cajas):
        correcto = posicion < len(esperado) and posicion < len(obtenido) and esperado[posicion] == obtenido[posicion]
        cv2.rectangle(color, (x, y), (x + w, y + h), (0, 160, 0) if correcto else (0, 0, 220), 1)
    for numero, (texto_esperado, texto_obtenido) in enumerate(zip(esperados, obtenidos)):
        cv2.putText(
            color,
            f"real {texto_esperado} | pred {texto_obtenido}",
            (10, 20 + numero * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return color


def demo_formulario(
    n_formularios: int = config.FORMULARIOS_DEMO,
    semilla: int = config.SEED,
    test_dir: Path = config.TEST_DIR,
    models_dir: Path = config.MODELS_DIR,
    metrics_dir: Path = config.METRICS_DIR,
    figures_dir: Path = config.FIGURES_DIR,
) -> dict[str, Any]:
    """Genera formularios, los lee con la CNN y mide acierto por dígito y por campo."""
    import keras

    ruta_modelo = Path(models_dir) / "cnn.keras"
    if not ruta_modelo.is_file():
        raise FileNotFoundError(f"Falta la CNN entrenada en {ruta_modelo}")
    modelo = keras.models.load_model(ruta_modelo)
    indice = _indice_test(test_dir)
    asegurar_directorio(figures_dir)

    aciertos_digitos = 0
    total_digitos = 0
    campos_exactos = 0
    total_campos = 0
    detalle = []

    for numero in range(n_formularios):
        rng = random.Random(semilla + numero)
        imagen, esperados = generar_formulario(indice, rng)
        obtenidos = leer_formulario(imagen, modelo)

        for posicion, esperado in enumerate(esperados):
            obtenido = obtenidos[posicion] if posicion < len(obtenidos) else ""
            total_campos += 1
            campos_exactos += int(esperado == obtenido)
            total_digitos += len(esperado)
            aciertos_digitos += sum(1 for a, b in zip(esperado, obtenido) if a == b)

        detalle.append(
            {
                "formulario": numero,
                "esperado": esperados,
                "obtenido": obtenidos,
                "campos_exactos": sum(1 for a, b in zip(esperados, obtenidos) if a == b),
            }
        )
        if numero < 3:
            cv2.imwrite(str(Path(figures_dir) / f"formulario_{numero:02d}.png"), _anotar(imagen, esperados, obtenidos))

    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "formularios": n_formularios,
        "campos_por_formulario": len(detalle[0]["esperado"]) if detalle else 0,
        "digitos_por_campo": config.DIGITOS_POR_CAMPO,
        "acierto_por_digito": round(aciertos_digitos / total_digitos, 4) if total_digitos else 0.0,
        "acierto_por_campo": round(campos_exactos / total_campos, 4) if total_campos else 0.0,
        "campos_exactos": campos_exactos,
        "total_campos": total_campos,
        "digitos_acertados": aciertos_digitos,
        "total_digitos": total_digitos,
        "detalle": detalle,
    }
    guardar_json(Path(metrics_dir) / "inferencia.json", resultado)
    logger.info(
        "Formularios: acierto por dígito=%.4f, campo exacto=%.4f (%d/%d)",
        resultado["acierto_por_digito"],
        resultado["acierto_por_campo"],
        campos_exactos,
        total_campos,
    )
    return resultado
