"""Punto de entrada del proyecto: ejecuta cada etapa en orden."""

from __future__ import annotations

import argparse
import logging

from src import config, entrenamiento, inferencia, utils
from src.datos import division, inspeccion, seleccion
from src.evaluacion import metricas, visualizacion
from src.modelos import clustering

logger = utils.obtener_logger("main")

ETAPAS = (
    "inspeccion",
    "seleccion",
    "division",
    "clasicos",
    "kmeans",
    "cnn",
    "evaluacion",
    "inferencia",
)


def parsear_argumentos() -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Pipeline de reconocimiento de dígitos de TecnoForms")
    parser.add_argument(
        "--etapas",
        nargs="+",
        choices=ETAPAS,
        default=list(ETAPAS),
        help="Etapas a ejecutar (por defecto, todas en orden)",
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Prueba de humo con una fracción pequeña de los datos",
    )
    parser.add_argument(
        "--muestras-cnn",
        type=int,
        default=None,
        help=f"Muestras de fit para entrenar la CNN (por defecto {config.CNN_SUBSAMPLE})",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=None,
        help="Épocas de la CNN (por defecto usa la configuración del proyecto)",
    )
    return parser.parse_args()


def main() -> None:
    """Orquesta las etapas seleccionadas."""
    args = parsear_argumentos()
    utils.configurar_logging()
    utils.fijar_semilla(config.SEED)
    total = len(args.etapas)

    for i, etapa in enumerate(args.etapas, start=1):
        logger.info("Etapa %d/%d: %s", i, total, etapa)
        if etapa == "inspeccion":
            inspeccion.generar_reporte(config.RAW_DIR, config.METRICS_DIR)
        elif etapa == "seleccion":
            seleccion.seleccionar(config.RAW_DIR, config.MANIFEST_PATH)
        elif etapa == "division":
            division.dividir(config.MANIFEST_PATH, config.FIT_DIR, config.TEST_DIR)
        elif etapa == "clasicos":
            entrenamiento.entrenar_clasicos(rapido=args.rapido)
        elif etapa == "kmeans":
            clustering.ejecutar(rapido=args.rapido)
        elif etapa == "cnn":
            entrenamiento.entrenar_cnn(
                rapido=args.rapido,
                muestras=args.muestras_cnn,
                epocas=args.epocas,
            )
        elif etapa == "evaluacion":
            resultados = metricas.evaluar_todos()
            visualizacion.generar_todas(resultados)
        elif etapa == "inferencia":
            inferencia.demo_formulario()

    logger.info("Proceso completado. Reportes en %s", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
