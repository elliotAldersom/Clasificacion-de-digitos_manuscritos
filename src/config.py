"""Constantes globales del proyecto (rutas, semilla, tamaños y presupuestos de cómputo)."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
RAW_DIR = BASE_DIR / "data" / "puro" / "by_class"
FIT_DIR = BASE_DIR / "data" / "fit"
TEST_DIR = BASE_DIR / "data" / "test"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.csv"

# Rutas de salida
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ARQUITECTURA_PATH = REPORTS_DIR / "arquitectura_cnn.txt"
INFORME_PATH = REPORTS_DIR / "informe.md"

# Registro de ejecución (no se versiona: se reescribe en cada corrida)
LOGS_DIR = BASE_DIR / "logs"
LOG_PATH = LOGS_DIR / "pipeline.log"

# Dígitos: códigos hexadecimales de NIST (30 → '0' ... 39 → '9')
DIGIT_CODES = [f"{codigo:02x}" for codigo in range(0x30, 0x3A)]
CLASES = list(range(10))
NOMBRES_MODELOS = {
    "cnn": "CNN",
    "svm": "SVM",
    "knn": "KNN",
    "regresion_logistica": "Regresión logística",
    "kmeans": "K-Means",
}

# Reproducibilidad y división
SEED = 42
FIT_RATIO = 0.80
TEST_RATIO = 0.20

# Imágenes
IMG_SIZE = 28
DIGIT_BOX = 20

# Entrenamiento
VAL_SPLIT = 0.10
CV_FOLDS = 5
CLASSIC_SUBSAMPLE = 10_000  # muestras de fit para KNN/SVM/regresión logística
CNN_SUBSAMPLE = 60_000  # muestras de fit para la CNN (se declara en el informe)
CNN_EPOCHS = 12
CNN_BATCH_SIZE = 128
CNN_LEARNING_RATE = 1e-3

# Segmentación
AREA_MINIMA = 30
TOLERANCIA_LINEA = 20

# Demo de formularios sintéticos
FORMULARIOS_DEMO = 10
DIGITOS_POR_CAMPO = 6

# Nombres de archivos de salida
MANIFEST_COLUMNS = [
    "ruta_original",
    "digito",
    "particion",
    "escritor",
    "hash",
    "valida",
    "motivo_descarte",
]
