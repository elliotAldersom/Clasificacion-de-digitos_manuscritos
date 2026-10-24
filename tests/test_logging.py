"""Pruebas del registro de eventos: salida a archivo e idempotencia de la configuración."""

from __future__ import annotations

from pathlib import Path

from src import utils


def _limpiar_logger() -> None:
    """Quita todos los manejadores para dejar el logger en estado limpio."""
    logger = utils.obtener_logger()
    for manejador in list(logger.handlers):
        logger.removeHandler(manejador)
        manejador.close()


def test_configurar_logging_escribe_en_archivo(tmp_path: Path) -> None:
    ruta = tmp_path / "logs" / "pipeline.log"
    _limpiar_logger()
    try:
        utils.configurar_logging(ruta_archivo=ruta)
        utils.obtener_logger("prueba").info("mensaje de prueba")
        assert ruta.exists()
        assert "mensaje de prueba" in ruta.read_text(encoding="utf-8")
    finally:
        _limpiar_logger()


def test_configurar_logging_no_duplica_manejadores(tmp_path: Path) -> None:
    ruta = tmp_path / "logs" / "pipeline.log"
    _limpiar_logger()
    try:
        utils.configurar_logging(ruta_archivo=ruta)
        utils.configurar_logging(ruta_archivo=ruta)
        assert len(utils.obtener_logger().handlers) == 2
    finally:
        _limpiar_logger()
