"""Descarga o generacion de los datos del caso aplicado, no se versionan en el repositorio."""
import os
from pathlib import Path

DIR_DATOS = Path(os.environ.get("PSI_DATOS", "data"))

URL_FOTOS = "Por agregar..."

def descargar_fotos(destino=DIR_DATOS):
    """Descarga las fotos con patron periodico real. Idempotente."""