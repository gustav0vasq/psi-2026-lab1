import zipfile

import pytest

from psi_lab.lab1 import datos


def _zip(ruta, entradas):
    with zipfile.ZipFile(ruta, "w") as z:
        for nombre, contenido in entradas.items():
            z.writestr(nombre, contenido)
    return ruta


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Apunta URL_FOTOS a un zip local (file://), sin red. Retorna una funcion para armarlo."""

    def armar(entradas):
        ruta = _zip(tmp_path / "imagenes.zip", entradas)
        monkeypatch.setattr(datos, "URL_FOTOS", ruta.as_uri())
        return ruta

    return armar


REQUERIDAS = {f"imagenes/{n}": b"x" for n in datos._FOTOS_REQUERIDAS}


def test_extrae_las_fotos_aunque_vengan_en_una_subcarpeta(release, tmp_path):
    release({**REQUERIDAS, "imagenes/otra.png": b"y", "imagenes/LEEME.txt": b"no es imagen"})
    destino = tmp_path / "data"
    caso = datos.descargar_fotos(destino)
    assert caso == destino / "caso_real"
    assert {p.name for p in caso.iterdir()} == {*datos._FOTOS_REQUERIDAS, "otra.png"}


def test_es_idempotente_no_vuelve_a_descargar(release, tmp_path):
    ruta = release(REQUERIDAS)
    destino = tmp_path / "data"
    datos.descargar_fotos(destino)
    ruta.unlink()  # si intentara descargar de nuevo, fallaria
    assert datos.descargar_fotos(destino) == destino / "caso_real"


def test_no_escribe_fuera_del_destino(release, tmp_path):
    release({**REQUERIDAS, "../../fuera.jpg": b"malicioso"})
    destino = tmp_path / "a" / "data"
    caso = datos.descargar_fotos(destino)
    assert (caso / "fuera.jpg").is_file()
    assert not (tmp_path / "fuera.jpg").exists()
    assert not (tmp_path / "a" / "fuera.jpg").exists()


def test_falla_con_mensaje_claro_si_faltan_fotos(release, tmp_path):
    release({"imagenes/diario.jpg": b"x"})
    with pytest.raises(RuntimeError, match="cielo.jpg"):
        datos.descargar_fotos(tmp_path / "data")


def test_falla_con_mensaje_claro_si_la_descarga_falla(tmp_path, monkeypatch):
    monkeypatch.setattr(datos, "URL_FOTOS", (tmp_path / "no_existe.zip").as_uri())
    with pytest.raises(RuntimeError, match="No se pudo descargar"):
        datos.descargar_fotos(tmp_path / "data")


def test_falla_con_mensaje_claro_si_no_es_un_zip(tmp_path, monkeypatch):
    falso = tmp_path / "falso.zip"
    falso.write_bytes(b"esto no es un zip")
    monkeypatch.setattr(datos, "URL_FOTOS", falso.as_uri())
    with pytest.raises(RuntimeError, match="no es un ZIP"):
        datos.descargar_fotos(tmp_path / "data")


def test_no_deja_archivos_temporales(release, tmp_path):
    release(REQUERIDAS)
    destino = tmp_path / "data"
    datos.descargar_fotos(destino)
    assert [p.name for p in destino.iterdir()] == ["caso_real"]
