# Reconocimiento de dígitos manuscritos (TecnoForms)

Feature: pipeline completo de ML + visión computacional sobre NIST SD19.
Rama: sin repositorio git (el directorio no está versionado; no se crea repo sin pedido explícito).
Runner: `pytest` (`.venv/bin/pytest`). Entorno: Python 3.12.15 standalone + TensorFlow CPU 2.21.

## Specs

- S1. El proyecto vive en la raíz actual con la estructura modular de `WORK.md` (`main.py`, `src/{datos,imagenes,modelos,evaluacion}`, `tests/`, `reports/`, `models/`).
- S2. `inspeccion` recorre `data/puro/by_class`, identifica carpetas `30`–`39` como dígitos, cuenta archivos por extensión, balance de clases, revisa los `.mit` (tamaño, cantidad, vista previa) y guarda `reports/metrics/inspeccion_dataset.json`.
- S3. `seleccion` conserva solo dígitos, imágenes legibles y con trazo, sin duplicados exactos (hash), y registra `data/manifest.csv` con `ruta_original, digito, hash, valida, motivo_descarte`.
- S4. `division` copia 80/20 estratificado por dígito a `data/fit/<d>/` y `data/test/<d>/`, reproducible, disjunto e idempotente.
- S5. `preprocesamiento` produce `28×28` `float32` en `[0,1]`, invierte fondo claro, recorta, escala (lado ≤ 20) y centra; error claro en imagen sin trazo.
- S6. `segmentacion` devuelve cajas de dígitos y las ordena en orden de lectura (izq→der, arriba→abajo).
- S7. `clasicos` implementa KNN, SVM y regresión logística en `Pipeline` con PCA.
- S8. `clustering` agrupa en 10 clústeres, reporta ARI, NMI y pureza, y guarda centroides.
- S9. `cnn` define la arquitectura convolucional y documenta `model.summary()` en `reports/arquitectura_cnn.txt`.
- S10. `entrenamiento` hace validación cruzada + búsqueda de hiperparámetros para clásicos y entrena la CNN con callbacks; registra `reports/metrics/hiperparametros.csv`.
- S11. `evaluacion` mide accuracy, precision/recall/F1 macro y por clase, matriz de confusión, tiempos y pares confundidos; genera `metricas_<modelo>.json` y `comparacion_modelos.csv`.
- S12. `visualizacion` genera muestras, distribución de clases, curvas, matrices, comparación, galerías y centroides en `reports/figures/`.
- S13. `inferencia` genera formularios sintéticos, segmenta, clasifica y reporta acierto por dígito y por campo.
- S14. `main.py` orquesta por etapas (`--etapas`, `--rapido`) sin lógica de negocio.
- S15. `tests/` cubre selección, división, preprocesamiento y segmentación.
- S16. `requirements.txt` con versiones reales, `.gitignore` completo, `README.md` completo y `reports/informe.md` con resultados reales.
- S17. `AGENTS.md` en la raíz con el contexto operativo para próximas sesiones.

## Tasks

- [x] T1. Config, utils, preprocesamiento y segmentación (inline) — S5, S6. Evidencia: `src/config.py`, `src/imagenes/*`, pruebas de preprocesamiento y segmentación en verde.
- [x] T2. Inspección, selección y división de datos (inline) — S2, S3, S4. Evidencia: `reports/metrics/inspeccion_dataset.json`, `data/manifest.csv` (747 260 filas), `reports/metrics/division.json` (402 951 válidas, 80 595 en test, `disjuntos: true`).
- [x] T3. Cargador con caché npz (inline) — S4, S5. Evidencia: `data/processed/fit_28x28.npz` (322 356), `test_28x28.npz` (80 595).
- [x] T4. Modelos clásicos, K-Means y CNN (inline) — S7, S8, S9. Evidencia: `models/{knn,svm,regresion_logistica}.joblib`, `models/kmeans_centroides.npz`, `models/cnn.keras`, `reports/arquitectura_cnn.txt`.
- [x] T5. Entrenamiento con CV/hiperparámetros (inline) — S10. Evidencia: `reports/metrics/hiperparametros.csv` (KNN 0.9560, SVM 0.9746, LR 0.9141); `entrenamiento_cnn.json` (base 0.9932 vs regularizada 0.9913).
- [x] T6. Evaluación, visualización e inferencia (inline) — S11, S12, S13. Evidencia: `comparacion_modelos.csv`, 4 `metricas_*.json`, 14 figuras, `inferencia.json` (85.83 % dígito, 55 % campo).
- [x] T7. `main.py`, pruebas y entorno (inline) — S14, S15. Evidencia: `.venv/bin/pytest` → 11 pruebas en verde; `--etapas` usado en corridas reales.
- [x] T8. Ejecución real del pipeline y revisión de métricas (inline). Evidencia: `logs/05_cnn.log`, `logs/06_eval.log`; accuracy CNN 0.9938 sobre 80 595.
- [x] T9. README, informe técnico y AGENTS.md con resultados reales — S16, S17. Evidencia: `README.md` (sección Resultados), `reports/informe.md` (19 secciones), `AGENTS.md`.

Ruta por tarea: todas directas inline (trabajo de un solo autor, superficie de edición contigua,
sin unidades paralelas independientes ni presión de contexto). Sin trigger de writer ni de
explorador. Revisión nativa (RDD) no aplicable: el directorio **no es un repositorio git**,
por lo que no existe candidato con commit base para `review assess`.

## Log

- L1. Pedido del usuario: "analiza WORK.md y realiza todo el trabajo indicado, no te olvides el agents.md para proximas sesiones."
- L2. Entorno detectado: Python del sistema 3.14.8 (sin wheels de TensorFlow), sin sudo; se instaló CPython 3.12.15 standalone en `~/.local/opt/py312` y se creó `.venv` con `tensorflow-cpu==2.21.0`.
- L3. `.mit` verificados: texto plano, primera línea = cantidad de entradas, luego `imagen_destino ruta_origen` (p. ej. `hsf_0_00000.png f0000_14/d0000_14/d0000_14_00000.png`). Se usa para trazabilidad (escritor `fNNNN`).
- L4. Imágenes NIST: PNG 128×128 RGB con fondo claro y trazo oscuro → el preprocesamiento invierte.
- L5. Decisión: división estratificada por dígito (texto del requisito 3); la limitación de fuga por escritor se documenta, con los IDs de escritor ya trazados en el manifiesto.
- L6. Selección ejecutada: 402 951 válidas, 344 309 duplicadas (344 307 de `train_*` + 2 colisiones internas de `hsf_*`), 0 descartes por ilegible o sin trazo. 3 579 escritores.
- L7. CNN entrenada en CPU: base 0.9932 `val_accuracy` (11/12 épocas, 436.9 s) vs regularizada 0.9913 (8 épocas, 319.2 s). Se elige `base` por `val_accuracy`; 422 026 parámetros.
- L8. Evaluación sobre 80 595 imágenes: CNN 0.9938, SVM 0.9773, KNN 0.9593, regresión logística 0.9200. Inferencia: formularios 85.83 % por dígito y 55 % por campo; 4 de los 9 campos fallidos son por segmentación (el acierto sube a 94.79 % en los 16 campos con longitud correcta).
- L9. Hallazgo de redacción: los pares confundidos coinciden en los cuatro modelos y en K-Means (9↔7, 9↔4, 5↔3, 8↔3/5, 3↔2), evidencia de que la confusión es del problema y no del clasificador.
- L10. Cierre: `README.md` con sección Resultados real, `reports/informe.md` con 19 secciones, `.venv/bin/pytest` en verde (11 pruebas). Directorio sin git, así que no hay commit de work unit ni revisión nativa.
