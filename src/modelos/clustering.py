"""Agrupamiento no supervisado de dígitos con K-Means."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src import config
from src.datos.cargador import cargar_directorio, cargar_fit, submuestra_estratificada
from src.utils import asegurar_directorio, guardar_json, obtener_logger

logger = obtener_logger("clustering")


def pureza_clusters(etiquetas_reales: np.ndarray, clusters: np.ndarray, n_clusters: int) -> dict[str, Any]:
    """Calcula la pureza de cada clúster y el dígito dominante."""
    detalle = []
    aciertos = 0
    for cluster in range(n_clusters):
        mascara = clusters == cluster
        total = int(mascara.sum())
        if total == 0:
            detalle.append({"cluster": cluster, "total": 0, "digito_dominante": None, "pureza": 0.0})
            continue
        valores, cuentas = np.unique(etiquetas_reales[mascara], return_counts=True)
        indice = int(np.argmax(cuentas))
        aciertos += int(cuentas[indice])
        detalle.append(
            {
                "cluster": cluster,
                "total": total,
                "digito_dominante": int(valores[indice]),
                "pureza": round(float(cuentas[indice] / total), 4),
            }
        )
    total_global = int(len(etiquetas_reales))
    return {
        "por_cluster": detalle,
        "pureza_global": round(aciertos / total_global, 4) if total_global else 0.0,
    }


def ejecutar(
    n_clusters: int = 10,
    n_muestras: int = 20_000,
    rapido: bool = False,
    fit_dir: Path = config.FIT_DIR,
) -> dict[str, Any]:
    """Agrupa ``data/fit`` sin etiquetas y compara los clústeres con los dígitos reales."""
    if rapido:
        n_muestras = 2_000
    X, y = cargar_fit() if Path(fit_dir) == config.FIT_DIR else cargar_directorio(fit_dir)
    X_sub, y_sub = submuestra_estratificada(X, y, n_muestras)
    plano = X_sub.reshape(len(X_sub), -1)

    logger.info("K-Means con %d clústeres sobre %d imágenes", n_clusters, len(plano))
    modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=config.SEED)
    clusters = modelo.fit_predict(plano)

    pureza = pureza_clusters(y_sub, clusters, n_clusters)
    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "n_muestras": int(len(plano)),
        "n_clusters": n_clusters,
        "ari": round(float(adjusted_rand_score(y_sub, clusters)), 4),
        "nmi": round(float(normalized_mutual_info_score(y_sub, clusters)), 4),
        "pureza_global": pureza["pureza_global"],
        "pureza_por_cluster": pureza["por_cluster"],
        "inercia": round(float(modelo.inertia_), 2),
    }

    asegurar_directorio(config.METRICS_DIR)
    guardar_json(config.METRICS_DIR / "kmeans.json", resultado)
    asegurar_directorio(config.MODELS_DIR)
    np.savez_compressed(
        config.MODELS_DIR / "kmeans_centroides.npz",
        centroides=modelo.cluster_centers_.reshape(n_clusters, config.IMG_SIZE, config.IMG_SIZE),
    )

    from src.evaluacion import visualizacion

    visualizacion.graficar_centroides(
        modelo.cluster_centers_.reshape(n_clusters, config.IMG_SIZE, config.IMG_SIZE),
        config.FIGURES_DIR / "kmeans_centroides.png",
        digitos_dominantes=[c["digito_dominante"] for c in pureza["por_cluster"]],
    )
    logger.info(
        "K-Means listo: ARI=%.4f NMI=%.4f pureza=%.4f",
        resultado["ari"],
        resultado["nmi"],
        resultado["pureza_global"],
    )
    return resultado
