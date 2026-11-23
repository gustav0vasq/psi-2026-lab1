import pytest

from psi_lab.lab1 import figuras


@pytest.fixture
def registro(monkeypatch):
    """Registro vacio, para no mezclar las figuras de prueba con las reales."""
    vacio = {}
    monkeypatch.setattr(figuras, "REGISTRO", vacio)
    monkeypatch.setattr(figuras, "descubrir", lambda: None)
    return vacio


def test_figura_registra_y_generar_la_ejecuta(registro, tmp_path):
    llamadas = []

    @figuras.figura("a")
    def _(salida):
        llamadas.append(salida)

    assert figuras.generar(tmp_path) == ["a"]
    assert llamadas == [tmp_path]


def test_sintetico_omite_las_que_requieren_datos(registro, tmp_path):
    llamadas = []
    figuras.figura("libre")(lambda salida: llamadas.append("libre"))
    figuras.figura("real", requiere_datos=True)(lambda salida: llamadas.append("real"))

    assert figuras.generar(tmp_path, sintetico=True) == ["libre"]
    assert llamadas == ["libre"]
    assert figuras.generar(tmp_path) == ["libre", "real"]


def test_nombre_duplicado_falla(registro):
    figuras.figura("a")(lambda salida: None)
    with pytest.raises(ValueError):
        figuras.figura("a")(lambda salida: None)


def test_error_en_una_figura_no_se_oculta(registro, tmp_path):
    def mala(salida):
        raise RuntimeError("fallo")

    figuras.figura("mala")(mala)
    with pytest.raises(RuntimeError):
        figuras.generar(tmp_path)


def test_guardar_tabla_escribe_csv_en_tablas(tmp_path):
    ruta = figuras.guardar_tabla(tmp_path, "t", ["f", "alias"], [[3, 3], [7, 3]])
    assert ruta == tmp_path / "tablas" / "t.csv"
    assert ruta.read_text().splitlines() == ["f,alias", "3,3", "7,3"]


def test_descubrir_encuentra_las_figuras_reales():
    figuras.descubrir()
    assert "muestreo_alias" in figuras.REGISTRO
