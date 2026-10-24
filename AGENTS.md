# AGENTS.md — Contexto operativo para próximas sesiones

Este archivo es la guía corta para retomar el proyecto sin releer todo. El detalle funcional
está en `WORK.md` (enunciado completo) y los resultados reales en `reports/`.

## Qué es el proyecto

Pipeline de reconocimiento de dígitos manuscritos para TecnoForms sobre NIST SD19
(`data/puro/by_class`): inspección, selección, división 80/20, preprocesamiento,
segmentación, CNN + KNN/SVM/regresión logística, K-Means, evaluación y lectura de
formularios sintéticos. Punto de entrada único: `main.py`.

## Entorno (leer antes de instalar nada)

- El Python del sistema es **3.14** y **no tiene wheels de TensorFlow**. No lo uses.
- El entorno vive en `.venv`, creado con un CPython **3.12.15 standalone** en
  `~/.local/opt/py312` (descargado de python-build-standalone porque no hay sudo).
- Si `.venv` se rompe: `~/.local/opt/py312/bin/python3.12 -m venv .venv` y
  `.venv/bin/pip install -r requirements.txt`.
- Todo se ejecuta con `.venv/bin/python` (o activando el entorno). TensorFlow es CPU-only
  (`tensorflow-cpu==2.21.0`); no hay GPU en esta máquina (12 núcleos, 15 GB RAM).

## Comandos

```bash
.venv/bin/python main.py                     # pipeline completo (~1 h con la config actual)
.venv/bin/python main.py --etapas inspeccion seleccion division
.venv/bin/python main.py --etapas cnn --muestras-cnn 30000 --epocas 8
.venv/bin/python main.py --rapido            # humo: 1 época, 2k muestras
.venv/bin/pytest                             # 13 pruebas, ~2 s, no tocan TensorFlow
```

Cada etapa lee y escribe en disco; se puede correr por separado. `configurar_logging()`
crea `logs/pipeline.log` y escribe ahí además de la consola, así que no hace falta
redirigir la salida a mano (el directorio no se versiona).

## Datos y artefactos

- `data/puro/` es **solo lectura**. Nunca se mueve, borra ni modifica.
- `data/fit/` y `data/test/` son copias (80/20) hechas por `division`; `division` las
  recrea desde cero, así que es idempotente pero **borra** lo que haya ahí.
- `data/manifest.csv`: una fila por imagen con dígito, partición (`hsf_*`/`train_*`),
  escritor, hash, validez y motivo de descarte. Es la fuente de verdad de la división.
- `data/processed/*.npz`: caché de arrays 28×28. **Si cambia la selección o el
  preprocesamiento, borrala** (`rm data/processed/*.npz`); la caché solo valida
  versión y tamaño, no el contenido de las imágenes.
- `models/`: `cnn.keras`, `cnn_<config>.keras`, `*.joblib`, `kmeans_centroides.npz`.
- `reports/metrics/`: JSON/CSV de inspección, división, hiperparámetros, métricas por
  modelo, comparación, K-Means e inferencia. `reports/figures/`: PNG.
- **Toda cifra del README o del informe sale de `reports/metrics/`.** No inventar números.

## Convenciones del código

- Un tema por archivo; `main.py` solo orquesta (no lógica de negocio).
- Nombres, docstrings y mensajes en **español neutro**; identificadores en español.
- Constantes en `src/config.py` (rutas, semilla, tamaños, presupuestos de cómputo).
- Rutas con `pathlib.Path`; logging con `src.utils.obtener_logger`, nunca `print`.
- Importar TensorFlow/Keras **dentro** de las funciones que lo usan (excepto
  `src/modelos/cnn.py`), para que los tests y las etapas de datos no paguen el arranque.
- Semilla fija: `utils.fijar_semilla(SEED, con_tensorflow=True)` antes de entrenar la CNN.

## Gotchas verificados

- Los `.mit` son texto: primera línea = cantidad de entradas; luego
  `imagen_destino ruta_origen` (p. ej. `hsf_0_00000.png f0000_14/d0000_14/...`).
  El prefijo `fNNNN` del origen identifica al escritor; se guarda en el manifiesto.
- Las imágenes NIST son PNG 128×128 **RGB con fondo claro**: el preprocesamiento las
  invierte (trazo claro sobre fondo oscuro).
- `train_*` contiene duplicados exactos de `hsf_*`; la deduplicación por hash los
  elimina antes de dividir (evita fuga de datos).
- La división es estratificada por dígito (requisito 3). La limitación de fuga por
  escritor está documentada en el informe; el manifiesto ya trae el escritor si se
  quiere migrar a `GroupShuffleSplit`.
- KNN/SVM usan submuestra (`CLASSIC_SUBSAMPLE`) y la CNN también (`CNN_SUBSAMPLE`):
  ambas se declaran en README e informe. Cambiarlas cambia los números.
- Segmentación: componentes conectados; `unir_fragmentos` recompone trazos partidos y
  no une dígitos vecinos. Dígitos que se tocan siguen siendo una limitación conocida.

## Estado y próximos pasos

Estado actual y resultados en `reports/informe.md` y `reports/metrics/`. El directorio
**no es un repositorio git**; si se decide versionar, `git init` y respetar `.gitignore`
(no subir `data/`, `models/`, `.venv/`).
