"""Inspección del dataset crudo (NIST SD19 ``by_class``) y de los archivos ``.mit``."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src import config
from src.utils import guardar_json, obtener_logger

logger = obtener_logger("inspeccion")


def clasificar_carpeta(nombre: str) -> int | None:
    """Devuelve el dígito de una carpeta hexadecimal ``30``–``39``, o ``None``."""
    try:
        codigo = int(nombre, 16)
    except ValueError:
        return None
    if nombre.lower() != f"{codigo:02x}":
        return None
    if 0x30 <= codigo <= 0x39:
        return codigo - 0x30
    return None


def contar_extensiones(carpeta: Path) -> Counter:
    """Cuenta archivos por extensión dentro de una carpeta (recursivo)."""
    return Counter(
        archivo.suffix.lower() for archivo in carpeta.rglob("*") if archivo.is_file()
    )


def leer_mit(ruta: Path) -> dict[str, str]:
    """Lee un ``.mit`` y devuelve ``{imagen_destino: ruta_origen}``.

    Formato comprobado en este dataset: primera línea con la cantidad de
    entradas y luego una línea por imagen ``destino origen``.
    """
    lineas = [linea.strip() for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
    if not lineas:
        return {}
    try:
        cantidad = int(lineas[0])
    except ValueError as error:
        raise ValueError(f"Primera línea no numérica en {ruta.name}") from error
    entradas = lineas[1:]
    if cantidad != len(entradas):
        raise ValueError(f"{ruta.name}: declara {cantidad} entradas pero tiene {len(entradas)}")
    mapa: dict[str, str] = {}
    for linea in entradas:
        partes = linea.split()
        if len(partes) >= 2:
            mapa[partes[0]] = partes[1]
    return mapa


def escritor_de_origen(origen: str) -> str:
    """Extrae el identificador de formulario/escritor (``f0000_14`` → ``f0000``)."""
    return origen.split("/")[0].split("_")[0]


def mapa_escritores(carpeta_digito: Path) -> dict[str, str]:
    """Construye ``{nombre_imagen: escritor}`` a partir de los ``.mit`` de la carpeta."""
    mapa: dict[str, str] = {}
    for mit in sorted(carpeta_digito.glob("*.mit")):
        for imagen, origen in leer_mit(mit).items():
            mapa[imagen] = escritor_de_origen(origen)
    return mapa


def vista_previa_mit(ruta: Path, n_bytes: int = 200) -> dict[str, Any]:
    """Describe un archivo ``.mit`` sin asumir que sea texto."""
    datos = ruta.read_bytes()[:n_bytes]
    try:
        contenido = datos.decode("utf-8")
        tipo = "texto"
    except UnicodeDecodeError:
        contenido = datos.hex(" ")
        tipo = "binario"
    return {
        "archivo": str(ruta),
        "tamano_bytes": ruta.stat().st_size,
        "tipo": tipo,
        "vista_previa": contenido,
    }


def analizar_mits(carpeta_digito: Path) -> dict[str, Any]:
    """Resume los ``.mit`` de una carpeta de dígito."""
    archivos = sorted(carpeta_digito.glob("*.mit"))
    if not archivos:
        return {"cantidad": 0, "tamano_medio_bytes": 0, "entradas": 0, "ejemplo": None}
    entradas = 0
    for archivo in archivos:
        try:
            entradas += len(leer_mit(archivo))
        except ValueError:
            entradas += 0
    return {
        "cantidad": len(archivos),
        "tamano_medio_bytes": int(np.mean([archivo.stat().st_size for archivo in archivos])),
        "entradas": entradas,
        "nombres": [archivo.name for archivo in archivos],
        "ejemplo": vista_previa_mit(archivos[0]),
    }


def muestra_imagenes(carpeta: Path, cantidad: int = 5) -> list[dict[str, Any]]:
    """Inspecciona tamaño, modo y rango de una muestra de imágenes."""
    muestra = []
    for ruta in sorted(carpeta.rglob("*.png"))[:cantidad]:
        imagen = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
        if imagen is None:
            muestra.append({"archivo": str(ruta), "legible": False})
            continue
        muestra.append(
            {
                "archivo": str(ruta),
                "legible": True,
                "forma": list(imagen.shape),
                "canales": 1 if imagen.ndim == 2 else int(imagen.shape[2]),
                "dtype": str(imagen.dtype),
                "min": int(imagen.min()),
                "max": int(imagen.max()),
                "media": round(float(imagen.mean()), 2),
            }
        )
    return muestra


def generar_reporte(raw_dir: Path = config.RAW_DIR, metrics_dir: Path = config.METRICS_DIR) -> dict[str, Any]:
    """Recorre ``by_class``, revisa los ``.mit`` y guarda el reporte en JSON."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de datos crudos: {raw_dir}")

    subcarpetas = sorted(carpeta for carpeta in raw_dir.iterdir() if carpeta.is_dir())
    por_clase: dict[str, Any] = {}
    balance: dict[str, int] = {}
    mits_total = 0
    mits_bytes = 0

    logger.info("Inspeccionando %s (%d subcarpetas)", raw_dir, len(subcarpetas))
    for carpeta in subcarpetas:
        digito = clasificar_carpeta(carpeta.name)
        extensiones = contar_extensiones(carpeta)
        info: dict[str, Any] = {
            "digito": digito,
            "es_digito": digito is not None,
            "extensiones": dict(sorted(extensiones.items())),
            "total_archivos": sum(extensiones.values()),
            "subcarpetas": sorted(hija.name for hija in carpeta.iterdir() if hija.is_dir()),
        }
        if digito is not None:
            mits = analizar_mits(carpeta)
            mits_total += mits["cantidad"]
            mits_bytes += mits["tamano_medio_bytes"] * mits["cantidad"]
            info["mits"] = mits
            info["muestra_imagenes"] = muestra_imagenes(carpeta)
            balance[str(digito)] = int(extensiones.get(".png", 0))
        por_clase[carpeta.name] = info

    valores = list(balance.values())
    reporte = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "raiz": str(raw_dir),
        "carpetas_totales": len(subcarpetas),
        "carpetas_digito": [carpeta.name for carpeta in subcarpetas if clasificar_carpeta(carpeta.name) is not None],
        "carpetas_otras": [carpeta.name for carpeta in subcarpetas if clasificar_carpeta(carpeta.name) is None],
        "por_clase": por_clase,
        "balance_png_por_digito": balance,
        "balance": {
            "total_png_digitos": sum(valores),
            "minimo": min(valores) if valores else 0,
            "maximo": max(valores) if valores else 0,
            "ratio_max_min": round(max(valores) / min(valores), 4) if valores and min(valores) else None,
        },
        "mits": {
            "total_archivos": mits_total,
            "tamano_aproximado_bytes": mits_bytes,
            "formato": (
                "Texto plano. Primera línea: cantidad de entradas. "
                "Luego una línea por imagen con 'imagen_destino ruta_origen' "
                "(p. ej. hsf_0_00000.png f0000_14/d0000_14/d0000_14_00000.png)."
            ),
            "uso_previsto": "Trazabilidad del escritor/formulario de cada imagen; no se usa como imagen de entrada.",
        },
    }
    ruta_salida = guardar_json(metrics_dir / "inspeccion_dataset.json", reporte)
    logger.info(
        "Inspección completa: %d imágenes de dígitos, %d .mit. Reporte en %s",
        sum(valores),
        mits_total,
        ruta_salida,
    )
    return reporte
