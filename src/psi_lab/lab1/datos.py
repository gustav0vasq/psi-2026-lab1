"""Descarga o generacion de los datos del caso aplicado, no se versionan en el repositorio."""
import os
from pathlib import Path

DIR_DATOS = Path(os.environ.get("PSI_DATOS", "data"))