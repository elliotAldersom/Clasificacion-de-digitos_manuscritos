"""Arquitectura de la red neuronal convolucional (modelo principal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
from keras import layers

from src import config
from src.utils import asegurar_directorio


def construir_cnn(
    input_shape: tuple[int, int, int] = (config.IMG_SIZE, config.IMG_SIZE, 1),
    n_clases: int = 10,
    filtros: tuple[int, int] = (32, 64),
    dropout_convolucional: float = 0.4,
    dropout_denso: float = 0.3,
    learning_rate: float = config.CNN_LEARNING_RATE,
) -> keras.Model:
    """Construye y compila la CNN de clasificación de dígitos."""
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(filtros[0], 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(filtros[1], 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(dropout_convolucional),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_denso),
            layers.Dense(n_clases, activation="softmax"),
        ],
        name="cnn_digitos",
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def guardar_arquitectura(
    modelo: keras.Model,
    ruta: Path = config.ARQUITECTURA_PATH,
    configuracion: dict[str, Any] | None = None,
) -> Path:
    """Guarda ``model.summary()`` y la configuración de entrenamiento en texto."""
    asegurar_directorio(ruta.parent)
    lineas = [
        "Arquitectura de la CNN de TecnoForms",
        "=" * 60,
        "",
    ]
    if configuracion:
        lineas.extend(f"{clave}: {valor}" for clave, valor in configuracion.items())
        lineas.extend(["", "Resumen de Keras", "-" * 60, ""])
    resumen: list[str] = []
    modelo.summary(print_fn=resumen.append)
    lineas.extend(resumen)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta
