"""Gráficos del proyecto: muestras, distribución, curvas, matrices, galerías y centroides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.datos.cargador import cargar_test
from src.imagenes.preprocesamiento import preprocesar_archivo
from src.utils import asegurar_directorio, cargar_json, obtener_logger

logger = obtener_logger("visualizacion")


def _guardar(fig: plt.Figure, ruta: Path) -> Path:
    asegurar_directorio(Path(ruta).parent)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(ruta)


def graficar_muestras(fit_dir: Path = config.FIT_DIR, ruta: Path = config.FIGURES_DIR / "muestras_dataset.png") -> Path:
    """Cuadrícula de un ejemplo por dígito, antes y después del preprocesamiento."""
    fig, ejes = plt.subplots(2, 10, figsize=(16, 3.8))
    for digito in range(10):
        carpeta = Path(fit_dir) / str(digito)
        archivos = sorted(carpeta.glob("*.png"))
        if not archivos:
            continue
        original = cv2.cvtColor(cv2.imread(str(archivos[0])), cv2.COLOR_BGR2RGB)
        procesada = preprocesar_archivo(archivos[0])
        ejes[0, digito].imshow(original)
        ejes[0, digito].set_title(f"original {digito}", fontsize=8)
        ejes[1, digito].imshow(procesada, cmap="gray")
        ejes[1, digito].set_title("28×28", fontsize=8)
        for eje in (ejes[0, digito], ejes[1, digito]):
            eje.axis("off")
    fig.suptitle("Muestras del dataset antes y después del preprocesamiento", fontsize=11)
    return _guardar(fig, ruta)


def graficar_distribucion(
    ruta_division: Path = config.METRICS_DIR / "division.json",
    ruta: Path = config.FIGURES_DIR / "distribucion_clases.png",
) -> Path:
    """Barras comparando la cantidad de imágenes por dígito en fit y test."""
    datos = cargar_json(ruta_division)["por_digito"]
    digitos = list(range(10))
    fit = [datos[str(d)]["fit"] for d in digitos]
    test = [datos[str(d)]["test"] for d in digitos]

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
    ancho = 0.4
    posiciones = np.arange(len(digitos))
    ejes[0].bar(posiciones - ancho / 2, fit, ancho, label="fit (80 %)", color="#4C72B0")
    ejes[0].bar(posiciones + ancho / 2, test, ancho, label="test (20 %)", color="#DD8452")
    ejes[0].set_xticks(posiciones, [str(d) for d in digitos])
    ejes[0].set_xlabel("Dígito")
    ejes[0].set_ylabel("Imágenes")
    ejes[0].set_title("Distribución de clases")
    ejes[0].legend()
    ejes[1].bar(posiciones, [datos[str(d)]["proporcion_test"] * 100 for d in digitos], color="#55A868")
    ejes[1].axhline(20, color="black", linestyle="--", linewidth=1)
    ejes[1].set_xticks(posiciones, [str(d) for d in digitos])
    ejes[1].set_ylim(0, 30)
    ejes[1].set_xlabel("Dígito")
    ejes[1].set_ylabel("% en test")
    ejes[1].set_title("Proporción en test por dígito")
    return _guardar(fig, ruta)


def graficar_curvas(
    historial: Path = config.METRICS_DIR / "cnn_historial.csv",
    ruta: Path = config.FIGURES_DIR / "curvas_entrenamiento.png",
) -> Path:
    """Curvas de loss y accuracy de la CNN (entrenamiento vs. validación)."""
    datos = pd.read_csv(historial)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    ejes[0].plot(datos["epoca"], datos["loss"], marker="o", label="entrenamiento")
    ejes[0].plot(datos["epoca"], datos["val_loss"], marker="o", label="validación")
    ejes[0].set_title("Pérdida (loss)")
    ejes[1].plot(datos["epoca"], datos["accuracy"], marker="o", label="entrenamiento")
    ejes[1].plot(datos["epoca"], datos["val_accuracy"], marker="o", label="validación")
    ejes[1].set_title("Exactitud (accuracy)")
    for eje in ejes:
        eje.set_xlabel("Época")
        eje.legend()
        eje.grid(alpha=0.3)
    return _guardar(fig, ruta)


def graficar_matriz(
    nombre: str,
    resultado: dict[str, Any],
    ruta: Path,
) -> Path:
    """Mapa de calor de la matriz de confusión normalizada por fila."""
    matriz = np.asarray(resultado["matriz_confusion"], dtype=float)
    normalizada = matriz / np.clip(matriz.sum(axis=1, keepdims=True), 1, None)
    fig, eje = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        normalizada,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Proporción de la clase real"},
        xticklabels=range(10),
        yticklabels=range(10),
        ax=eje,
    )
    eje.set_xlabel("Predicho")
    eje.set_ylabel("Real")
    eje.set_title(f"Matriz de confusión — {resultado['nombre']}")
    return _guardar(fig, ruta)


def graficar_comparacion(
    comparacion: Path = config.METRICS_DIR / "comparacion_modelos.csv",
    ruta: Path = config.FIGURES_DIR / "comparacion_modelos.png",
) -> Path:
    """Barras de accuracy y F1 macro por modelo."""
    datos = pd.read_csv(comparacion)
    posiciones = np.arange(len(datos))
    ancho = 0.38
    fig, eje = plt.subplots(figsize=(9, 4.5))
    eje.bar(posiciones - ancho / 2, datos["accuracy"], ancho, label="Accuracy", color="#4C72B0")
    eje.bar(posiciones + ancho / 2, datos["f1_macro"], ancho, label="F1 macro", color="#DD8452")
    eje.set_xticks(posiciones, datos["modelo"], rotation=15)
    eje.set_ylim(0, 1.05)
    eje.set_ylabel("Puntaje")
    eje.set_title("Comparación de modelos sobre data/test")
    for posicion, valor in zip(posiciones - ancho / 2, datos["accuracy"]):
        eje.text(posicion, valor + 0.01, f"{valor:.3f}", ha="center", fontsize=8)
    for posicion, valor in zip(posiciones + ancho / 2, datos["f1_macro"]):
        eje.text(posicion, valor + 0.01, f"{valor:.3f}", ha="center", fontsize=8)
    eje.legend(loc="lower right")
    return _guardar(fig, ruta)


def galeria_predicciones(
    X: np.ndarray,
    y_real: np.ndarray,
    y_pred: np.ndarray,
    ruta: Path = config.FIGURES_DIR / "galeria_predicciones.png",
    cantidad: int = 25,
) -> Path:
    """Cuadrícula de predicciones de la CNN (verde acierto, rojo error)."""
    indices = np.arange(min(cantidad, len(y_real)))
    fig, ejes = plt.subplots(5, 5, figsize=(9, 9))
    for eje, indice in zip(ejes.ravel(), indices):
        eje.imshow(X[indice], cmap="gray")
        real, pred = int(y_real[indice]), int(y_pred[indice])
        eje.set_title(
            f"real {real} | pred {pred}",
            color="green" if real == pred else "red",
            fontsize=9,
        )
        eje.axis("off")
    fig.suptitle("Galería de predicciones de la CNN")
    return _guardar(fig, ruta)


def galeria_errores(
    X: np.ndarray,
    y_real: np.ndarray,
    y_pred: np.ndarray,
    probabilidad: np.ndarray,
    ruta: Path = config.FIGURES_DIR / "galeria_errores.png",
    cantidad: int = 25,
) -> Path:
    """Los errores de la CNN con mayor confianza."""
    errores = np.flatnonzero(y_real != y_pred)
    if errores.size == 0:
        errores = np.arange(len(y_real))
    ordenados = errores[np.argsort(probabilidad[errores])[::-1]][:cantidad]
    fig, ejes = plt.subplots(5, 5, figsize=(9, 9))
    for eje, indice in zip(ejes.ravel(), ordenados):
        eje.imshow(X[indice], cmap="gray")
        eje.set_title(
            f"real {int(y_real[indice])} | pred {int(y_pred[indice])}\nconf {probabilidad[indice]:.2f}",
            color="red",
            fontsize=8,
        )
        eje.axis("off")
    for eje in ejes.ravel()[len(ordenados):]:
        eje.axis("off")
    fig.suptitle("Errores más confiados de la CNN")
    return _guardar(fig, ruta)


def graficar_centroides(
    centroides: np.ndarray,
    ruta: Path = config.FIGURES_DIR / "kmeans_centroides.png",
    digitos_dominantes: Sequence[int | None] | None = None,
) -> Path:
    """Centroides de K-Means vistos como imágenes de 28×28."""
    n = len(centroides)
    fig, ejes = plt.subplots(2, (n + 1) // 2, figsize=(2 * ((n + 1) // 2), 4.4))
    for indice, eje in enumerate(np.ravel(ejes)):
        eje.axis("off")
        if indice >= n:
            continue
        eje.imshow(centroides[indice], cmap="gray")
        titulo = f"clúster {indice}"
        if digitos_dominantes is not None and digitos_dominantes[indice] is not None:
            titulo += f"\ndígito {digitos_dominantes[indice]}"
        eje.set_title(titulo, fontsize=9)
    fig.suptitle("Centroides de K-Means")
    return _guardar(fig, ruta)


def generar_todas(
    resultados: dict[str, Any] | None = None,
    figuras_dir: Path = config.FIGURES_DIR,
) -> list[Path]:
    """Genera todas las figuras posibles a partir de los artefactos en disco."""
    asegurar_directorio(figuras_dir)
    if resultados is None:
        ruta = config.METRICS_DIR / "evaluacion.json"
        if not ruta.is_file():
            raise FileNotFoundError("Falta reports/metrics/evaluacion.json; ejecutá la etapa evaluacion")
        resultados = cargar_json(ruta)

    generadas: list[Path] = []
    if Path(config.FIT_DIR).is_dir():
        generadas.append(graficar_muestras(ruta=figuras_dir / "muestras_dataset.png"))
    if (config.METRICS_DIR / "division.json").is_file():
        generadas.append(graficar_distribucion(ruta=figuras_dir / "distribucion_clases.png"))
    if (config.METRICS_DIR / "cnn_historial.csv").is_file():
        generadas.append(graficar_curvas(ruta=figuras_dir / "curvas_entrenamiento.png"))

    for nombre, resultado in resultados.get("modelos", {}).items():
        generadas.append(
            graficar_matriz(nombre, resultado, figuras_dir / f"matriz_confusion_{nombre}.png")
        )

    if (config.METRICS_DIR / "comparacion_modelos.csv").is_file():
        generadas.append(graficar_comparacion(ruta=figuras_dir / "comparacion_modelos.png"))

    if (config.METRICS_DIR / "predicciones_cnn.npz").is_file():
        X, _ = cargar_test()
        datos = np.load(config.METRICS_DIR / "predicciones_cnn.npz")
        generadas.append(
            galeria_predicciones(X, datos["y_real"], datos["y_pred"], figuras_dir / "galeria_predicciones.png")
        )
        generadas.append(
            galeria_errores(
                X,
                datos["y_real"],
                datos["y_pred"],
                datos["probabilidad_maxima"],
                figuras_dir / "galeria_errores.png",
            )
        )

    centroides = config.MODELS_DIR / "kmeans_centroides.npz"
    if centroides.is_file():
        digitos = None
        ruta_kmeans = config.METRICS_DIR / "kmeans.json"
        if ruta_kmeans.is_file():
            digitos = [c["digito_dominante"] for c in cargar_json(ruta_kmeans)["pureza_por_cluster"]]
        generadas.append(
            graficar_centroides(
                np.load(centroides)["centroides"],
                figuras_dir / "kmeans_centroides.png",
                digitos_dominantes=digitos,
            )
        )

    logger.info("Figuras generadas: %d en %s", len(generadas), figuras_dir)
    return generadas
