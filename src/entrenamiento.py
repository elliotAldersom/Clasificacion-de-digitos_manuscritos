"""Entrenamiento de los modelos clásicos y de la CNN, con validación y hiperparámetros."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import config
from src.datos.cargador import a_entrada_cnn, cargar_fit, submuestra_estratificada
from src.modelos import clasicos, cnn
from src.utils import asegurar_directorio, fijar_semilla, guardar_json, obtener_logger

logger = obtener_logger("entrenamiento")

CSV_HIPERPARAMETROS = "hiperparametros.csv"


def _registrar_hiperparametros(filas: list[dict[str, Any]], metrics_dir: Path) -> Path:
    """Agrega o reemplaza filas de un modelo/etapa en ``hiperparametros.csv``."""
    asegurar_directorio(metrics_dir)
    ruta = Path(metrics_dir) / CSV_HIPERPARAMETROS
    nuevas = pd.DataFrame(filas)
    if ruta.is_file():
        previas = pd.read_csv(ruta)
        mascara = ~(
            previas["modelo"].isin(nuevas["modelo"]) & previas["etapa"].isin(nuevas["etapa"])
        )
        combinadas = pd.concat([previas[mascara], nuevas], ignore_index=True)
    else:
        combinadas = nuevas
    combinadas.to_csv(ruta, index=False)
    return ruta


def entrenar_clasicos(
    rapido: bool = False,
    fit_dir: Path = config.FIT_DIR,
    models_dir: Path = config.MODELS_DIR,
    metrics_dir: Path = config.METRICS_DIR,
) -> dict[str, Any]:
    """Valida, busca hiperparámetros y entrena KNN, SVM y regresión logística."""
    fijar_semilla(config.SEED)
    cantidad = 2_000 if rapido else config.CLASSIC_SUBSAMPLE
    X, y = cargar_fit()
    X_sub, y_sub = submuestra_estratificada(X, y, cantidad)
    X_plano = X_sub.reshape(len(X_sub), -1)
    logger.info(
        "Modelos clásicos: %d muestras de fit (%d por dígito en promedio)",
        len(y_sub),
        len(y_sub) // 10,
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}
    filas: list[dict[str, Any]] = []
    resumen: dict[str, Any] = {}
    asegurar_directorio(models_dir)

    for nombre, pipeline in clasicos.construir_modelos().items():
        rejilla = clasicos.rejillas_hiperparametros(rapido=rapido)[nombre]
        logger.info("Buscando hiperparámetros de %s: %s", nombre, rejilla)
        busqueda = GridSearchCV(
            estimator=pipeline,
            param_grid=rejilla,
            cv=cv,
            scoring=scoring,
            refit="f1_macro",
            n_jobs=-1,
            return_train_score=False,
        )
        inicio = time.perf_counter()
        busqueda.fit(X_plano, y_sub)
        duracion = time.perf_counter() - inicio

        indice = int(busqueda.best_index_)
        fila = {
            "etapa": "clasico",
            "modelo": nombre,
            "muestras": int(len(y_sub)),
            "rejilla": json.dumps(rejilla, ensure_ascii=False),
            "mejor_config": json.dumps(busqueda.best_params_, ensure_ascii=False),
            "f1_macro_cv": round(float(busqueda.cv_results_["mean_test_f1_macro"][indice]), 4),
            "accuracy_cv": round(float(busqueda.cv_results_["mean_test_accuracy"][indice]), 4),
            "tiempo_segundos": round(duracion, 1),
            "epocas": None,
        }
        filas.append(fila)
        ruta_modelo = clasicos.guardar_modelo(busqueda.best_estimator_, nombre, models_dir)
        resumen[nombre] = {
            **fila,
            "archivo": str(ruta_modelo),
            "pliegues": config.CV_FOLDS,
        }
        logger.info(
            "%s listo: mejor=%s f1_macro_cv=%.4f accuracy_cv=%.4f (%.1fs)",
            nombre,
            busqueda.best_params_,
            fila["f1_macro_cv"],
            fila["accuracy_cv"],
            duracion,
        )

    _registrar_hiperparametros(filas, metrics_dir)
    guardar_json(
        Path(metrics_dir) / "entrenamiento_clasicos.json",
        {"fecha": datetime.now().isoformat(timespec="seconds"), "resultados": resumen},
    )
    return resumen


def _configuraciones_cnn(rapido: bool) -> list[dict[str, Any]]:
    if rapido:
        return [
            {
                "nombre": "rapida",
                "learning_rate": 1e-3,
                "dropout_convolucional": 0.4,
                "dropout_denso": 0.3,
                "filtros": (32, 64),
                "epocas": 1,
            }
        ]
    return [
        {
            "nombre": "base",
            "learning_rate": 1e-3,
            "dropout_convolucional": 0.4,
            "dropout_denso": 0.3,
            "filtros": (32, 64),
            "epocas": config.CNN_EPOCHS,
        },
        {
            "nombre": "regularizada",
            "learning_rate": 5e-4,
            "dropout_convolucional": 0.5,
            "dropout_denso": 0.4,
            "filtros": (32, 64),
            "epocas": max(4, config.CNN_EPOCHS - 4),
        },
    ]


def entrenar_cnn(
    rapido: bool = False,
    muestras: int | None = None,
    epocas: int | None = None,
    models_dir: Path = config.MODELS_DIR,
    metrics_dir: Path = config.METRICS_DIR,
) -> dict[str, Any]:
    """Entrena la CNN comparando pocas configuraciones y guarda la mejor."""
    fijar_semilla(config.SEED, con_tensorflow=True)
    cantidad = 2_000 if rapido else (muestras or config.CNN_SUBSAMPLE)
    X, y = cargar_fit()
    X_sub, y_sub = submuestra_estratificada(X, y, cantidad)
    X_entrada = a_entrada_cnn(X_sub)
    X_train, X_val, y_train, y_val = train_test_split(
        X_entrada,
        y_sub,
        test_size=config.VAL_SPLIT,
        random_state=config.SEED,
        stratify=y_sub,
    )
    logger.info(
        "CNN: %d para entrenar, %d para validar (%d muestras de fit)",
        len(y_train),
        len(y_val),
        len(y_sub),
    )

    asegurar_directorio(models_dir)
    mejor_archivo = Path(models_dir) / "cnn.keras"
    filas: list[dict[str, Any]] = []
    historiales: dict[str, pd.DataFrame] = {}
    resumen: dict[str, Any] = {}

    for configuracion in _configuraciones_cnn(rapido):
        numero_epocas = epocas or configuracion["epocas"]
        modelo = cnn.construir_cnn(
            filtros=configuracion["filtros"],
            dropout_convolucional=configuracion["dropout_convolucional"],
            dropout_denso=configuracion["dropout_denso"],
            learning_rate=configuracion["learning_rate"],
        )
        ruta_config = Path(models_dir) / f"cnn_{configuracion['nombre']}.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(ruta_config), monitor="val_accuracy", save_best_only=True, verbose=0
            ),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoca, logs: logger.info(
                    "  %s época %d: val_accuracy=%.4f val_loss=%.4f",
                    configuracion["nombre"],
                    epoca + 1,
                    logs.get("val_accuracy", float("nan")),
                    logs.get("val_loss", float("nan")),
                )
            ),
        ]
        logger.info(
            "Entrenando configuración '%s' (%d épocas, lr=%s, dropout=%s/%s)",
            configuracion["nombre"],
            numero_epocas,
            configuracion["learning_rate"],
            configuracion["dropout_convolucional"],
            configuracion["dropout_denso"],
        )
        inicio = time.perf_counter()
        historial = modelo.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=numero_epocas,
            batch_size=config.CNN_BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )
        duracion = time.perf_counter() - inicio
        datos = pd.DataFrame(historial.history)
        datos.insert(0, "epoca", np.arange(1, len(datos) + 1))
        historiales[configuracion["nombre"]] = datos
        mejor_val = float(datos["val_accuracy"].max())
        fila = {
            "etapa": "cnn",
            "modelo": f"cnn_{configuracion['nombre']}",
            "muestras": int(len(y_train)),
            "rejilla": json.dumps(
                {
                    "learning_rate": configuracion["learning_rate"],
                    "dropout_convolucional": configuracion["dropout_convolucional"],
                    "dropout_denso": configuracion["dropout_denso"],
                    "filtros": list(configuracion["filtros"]),
                },
                ensure_ascii=False,
            ),
            "mejor_config": json.dumps({"epocas": numero_epocas}, ensure_ascii=False),
            "f1_macro_cv": None,
            "accuracy_cv": round(mejor_val, 4),
            "tiempo_segundos": round(duracion, 1),
            "epocas": int(len(datos)),
        }
        filas.append(fila)
        resumen[configuracion["nombre"]] = {
            **fila,
            "archivo": str(ruta_config),
            "val_loss_min": round(float(datos["val_loss"].min()), 4),
        }
        logger.info(
            "Configuración '%s': mejor val_accuracy=%.4f en %.1fs",
            configuracion["nombre"],
            mejor_val,
            duracion,
        )

    mejor_nombre = max(resumen, key=lambda clave: resumen[clave]["accuracy_cv"])
    mejor_config_path = Path(resumen[mejor_nombre]["archivo"])
    modelo_final = keras.models.load_model(mejor_config_path)
    modelo_final.save(mejor_archivo)
    historiales[mejor_nombre].to_csv(Path(metrics_dir) / "cnn_historial.csv", index=False)

    entrenamientos = {c["nombre"]: c for c in _configuraciones_cnn(rapido)}
    elegida = entrenamientos[mejor_nombre]
    arquitectura = cnn.guardar_arquitectura(
        modelo_final,
        config.ARQUITECTURA_PATH,
        configuracion={
            "configuración elegida": mejor_nombre,
            "muestras de entrenamiento": len(y_train),
            "muestras de validación": len(y_val),
            "épocas ejecutadas": resumen[mejor_nombre]["epocas"],
            "batch_size": config.CNN_BATCH_SIZE,
            "filtros por capa convolucional": list(elegida["filtros"]),
            "dropout convolucional / denso": f"{elegida['dropout_convolucional']} / {elegida['dropout_denso']}",
            "optimizador": "Adam",
            "learning_rate": elegida["learning_rate"],
            "loss": "sparse_categorical_crossentropy",
        },
    )
    _registrar_hiperparametros(filas, metrics_dir)
    guardar_json(
        Path(metrics_dir) / "entrenamiento_cnn.json",
        {
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "mejor": mejor_nombre,
            "modelo": str(mejor_archivo),
            "arquitectura": str(arquitectura),
            "configuraciones": resumen,
        },
    )
    logger.info("CNN lista: mejor configuración '%s' → %s", mejor_nombre, mejor_archivo)
    return resumen
