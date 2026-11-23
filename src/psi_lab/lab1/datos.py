"""Descarga o generacion de los datos del caso aplicado, no se versionan en el repositorio."""
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DIR_DATOS = Path(os.environ.get("PSI_DATOS", "data"))

URL_FOTOS = (
    "https://github.com/gustav0vasq/psi-2026-lab1/releases/download/"
    "v0.1.0-alpha/imagenes.zip"
)

# Las fotos que cargan las figuras del caso real (figuras/fig_caso_real.py)
_FOTOS_REQUERIDAS = ("diario.jpg", "cielo.jpg")
_TIMEOUT = 30  # [s] por operacion de red
_EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def descargar_fotos(destino=DIR_DATOS):
    """Descarga las fotos con patron periodico real. Idempotente.

    Las imágenes se guardan en ``destino/caso_real`` y se retorna esa ruta. Si todas las fotos
    requeridas ya existen, no se realiza ninguna descarga. Lanza RuntimeError con un mensaje
    claro si la descarga falla, si el archivo no es un ZIP o si faltan fotos requeridas.
    """
    destino = Path(destino)
    destino_caso = destino / "caso_real"

    if all((destino_caso / nombre).is_file() for nombre in _FOTOS_REQUERIDAS):
        return destino_caso

    destino.mkdir(parents=True, exist_ok=True)

    archivo_zip = None
    try:
        descriptor, archivo_temporal = tempfile.mkstemp(suffix=".zip", dir=destino)
        os.close(descriptor)
        archivo_zip = Path(archivo_temporal)
        try:
            with (
                urllib.request.urlopen(URL_FOTOS, timeout=_TIMEOUT) as respuesta,
                archivo_zip.open("wb") as salida,
            ):
                shutil.copyfileobj(respuesta, salida)
        except OSError as error:  # URLError, HTTPError y timeouts
            raise RuntimeError(f"No se pudo descargar {URL_FOTOS}: {error}") from error

        destino_caso.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archivo_zip) as zip_file:
                for entrada in zip_file.infolist():
                    if entrada.is_dir():
                        continue

                    # Solo el nombre del archivo: las fotos quedan planas y nada sale de destino
                    nombre = Path(entrada.filename).name
                    if not nombre or Path(nombre).suffix.lower() not in _EXTENSIONES_IMAGEN:
                        continue

                    with (
                        zip_file.open(entrada) as fuente,
                        (destino_caso / nombre).open("wb") as salida,
                    ):
                        shutil.copyfileobj(fuente, salida)
        except zipfile.BadZipFile as error:
            raise RuntimeError(f"{URL_FOTOS} no es un ZIP valido") from error
    finally:
        if archivo_zip is not None:
            archivo_zip.unlink(missing_ok=True)

    faltantes = [
        nombre for nombre in _FOTOS_REQUERIDAS
        if not (destino_caso / nombre).is_file()
    ]
    if faltantes:
        raise RuntimeError(
            "El ZIP no contiene las fotos requeridas: " + ", ".join(faltantes)
        )

    return destino_caso