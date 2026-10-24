"""Evaluación con el conjunto de prueba: métricas, matrices de confusión y comparación."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config
from src.datos.cargador import a_entrada_cnn, cargar_test
from src.modelos import clasicos
from src.utils import asegurar_directorio, cargar_json, guardar_json, obtener_logger

logger = obtener_logger("metricas")

ORDEN_MODELOS = ["cnn", "svm", "knn", "regresion_logistica"]


def pares_confundidos(matriz: np.ndarray, limite: int = 5) -> list[dict[str, Any]]:
    """Devuelve los pares (real, predicho) con más confusiones, sin contar la diagonal."""
    pares = [
        {
            "real": int(i),
            "predicho": int(j),
            "casos": int(matriz[i, j]),
            "porcentaje_de_la_clase": round(100 * matriz[i, j] / matriz[i].sum(), 2)
            if matriz[i].sum()
            else 0.0,
        }
        for i in range(matriz.shape[0])
        for j in range(matriz.shape[1])
        if i != j and matriz[i, j] > 0
    ]
    return sorted(pares, key=lambda par: par["casos"], reverse=True)[:limite]


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Calcula el bloque completo de métricas para un modelo."""
    etiquetas = list(range(10))
    matriz = confusion_matrix(y_real, y_pred, labels=etiquetas)
    reporte = classification_report(
        y_real, y_pred, labels=etiquetas, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_real, y_pred, average="weighted", zero_division=0)), 4),
        "por_clase": {
            str(clase): {
                "precision": round(float(reporte[str(clase)]["precision"]), 4),
                "recall": round(float(reporte[str(clase)]["recall"]), 4),
                "f1": round(float(reporte[str(clase)]["f1-score"]), 4),
                "soporte": int(reporte[str(clase)]["support"]),
            }
            for clase in etiquetas
        },
        "matriz_confusion": matriz.tolist(),
        "pares_confundidos": pares_confundidos(matriz),
    }


def _tiempo_entrenamiento(nombre: str, metrics_dir: Path) -> float | None:
    """Recupera el tiempo de entrenamiento registrado para un modelo."""
    ruta = Path(metrics_dir) / "entrenamiento_clasicos.json"
    if ruta.is_file() and nombre != "cnn":
        datos = cargar_json(ruta)
        return datos.get("resultados", {}).get(nombre, {}).get("tiempo_segundos")
    ruta_cnn = Path(metrics_dir) / "entrenamiento_cnn.json"
    if ruta_cnn.is_file() and nombre == "cnn":
        datos = cargar_json(ruta_cnn)
        mejor = datos.get("mejor")
        return datos.get("configuraciones", {}).get(mejor, {}).get("tiempo_segundos")
    return None


def _accuracy_validacion(nombre: str, metrics_dir: Path) -> float | None:
    """Accuracy de validación/cruzada registrada durante el entrenamiento."""
    if nombre == "cnn":
        ruta = Path(metrics_dir) / "entrenamiento_cnn.json"
        if ruta.is_file():
            datos = cargar_json(ruta)
            mejor = datos.get("mejor")
            return datos.get("configuraciones", {}).get(mejor, {}).get("accuracy_cv")
        return None
    ruta = Path(metrics_dir) / "entrenamiento_clasicos.json"
    if ruta.is_file():
        datos = cargar_json(ruta)
        return datos.get("resultados", {}).get(nombre, {}).get("accuracy_cv")
    return None


def evaluar_todos(
    test_dir: Path = config.TEST_DIR,
    models_dir: Path = config.MODELS_DIR,
    metrics_dir: Path = config.METRICS_DIR,
) -> dict[str, Any]:
    """Evalúa cada modelo disponible con ``data/test`` (una sola vez) y compara."""
    X_test, y_test = cargar_test() if Path(test_dir) == config.TEST_DIR else cargar_test()
    X_plano = X_test.reshape(len(X_test), -1)
    X_cnn = a_entrada_cnn(X_test)
    asegurar_directorio(metrics_dir)
    logger.info("Evaluando %d imágenes de prueba", len(y_test))

    predicciones: dict[str, Callable[[], np.ndarray]] = {}
    probabilidades: dict[str, np.ndarray] = {}

    ruta_cnn = Path(models_dir) / "cnn.keras"
    if ruta_cnn.is_file():
        import keras

        modelo_cnn = keras.models.load_model(ruta_cnn)

        def predecir_cnn() -> np.ndarray:
            salida = modelo_cnn.predict(X_cnn, batch_size=512, verbose=0)
            probabilidades["cnn"] = salida.max(axis=1)
            return salida.argmax(axis=1)

        predicciones["cnn"] = predecir_cnn
    else:
        logger.warning("No se encontró %s; se omite la CNN", ruta_cnn)

    for nombre in ORDEN_MODELOS:
        if nombre == "cnn":
            continue
        try:
            modelo = clasicos.cargar_modelo(nombre, models_dir)
        except FileNotFoundError:
            logger.warning("No se encontró el modelo '%s'; se omite", nombre)
            continue
        predicciones[nombre] = lambda modelo=modelo: modelo.predict(X_plano)

    if not predicciones:
        raise FileNotFoundError("No hay modelos entrenados para evaluar")

    resultados: dict[str, Any] = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "test": {
            "imagenes": int(len(y_test)),
            "por_digito": {str(int(d)): int((y_test == d).sum()) for d in range(10)},
        },
        "modelos": {},
        "comparacion": [],
    }
    predicciones_guardadas: dict[str, np.ndarray] = {}

    for nombre in ORDEN_MODELOS:
        if nombre not in predicciones:
            continue
        inicio = time.perf_counter()
        y_pred = predicciones[nombre]()
        duracion = time.perf_counter() - inicio
        predicciones_guardadas[nombre] = y_pred
        metricas = calcular_metricas(y_test, y_pred)
        tiempo_entrenamiento = _tiempo_entrenamiento(nombre, metrics_dir)
        metadatos = {
            "nombre": config.NOMBRES_MODELOS.get(nombre, nombre),
            "imagenes_evaluadas": int(len(y_test)),
            "tiempo_inferencia_total_segundos": round(duracion, 2),
            "tiempo_inferencia_por_imagen_ms": round(1000 * duracion / len(y_test), 3),
            "tiempo_entrenamiento_segundos": tiempo_entrenamiento,
            "accuracy_validacion": _accuracy_validacion(nombre, metrics_dir),
        }
        resultado = {**metadatos, **metricas}
        resultados["modelos"][nombre] = resultado
        guardar_json(Path(metrics_dir) / f"metricas_{nombre}.json", resultado)
        resultados["comparacion"].append(
            {
                "modelo": metadatos["nombre"],
                "accuracy": metricas["accuracy"],
                "precision_macro": metricas["precision_macro"],
                "recall_macro": metricas["recall_macro"],
                "f1_macro": metricas["f1_macro"],
                "accuracy_validacion": metadatos["accuracy_validacion"],
                "entrenamiento_segundos": tiempo_entrenamiento,
                "inferencia_por_imagen_ms": metadatos["tiempo_inferencia_por_imagen_ms"],
                "imagenes_evaluadas": metadatos["imagenes_evaluadas"],
            }
        )
        logger.info(
            "%s: accuracy=%.4f f1_macro=%.4f (%.2f ms/imagen)",
            metadatos["nombre"],
            metricas["accuracy"],
            metricas["f1_macro"],
            metadatos["tiempo_inferencia_por_imagen_ms"],
        )

    if "cnn" in probabilidades:
        np.savez_compressed(
            Path(metrics_dir) / "predicciones_cnn.npz",
            y_real=y_test,
            y_pred=predicciones_guardadas["cnn"],
            probabilidad_maxima=probabilidades["cnn"],
        )

    comparacion = pd.DataFrame(resultados["comparacion"])
    comparacion.to_csv(Path(metrics_dir) / "comparacion_modelos.csv", index=False)
    guardar_json(Path(metrics_dir) / "evaluacion.json", resultados)
    return resultados
