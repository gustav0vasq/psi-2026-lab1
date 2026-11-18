"""C2: coseno 2D, interferencia y deteccion de picos. Solo imagenes sinteticas (CI)."""

import numpy as np
import pytest

from psi_lab.lab1.caso_imagen import agregar_interferencia, coseno_2d, detectar_picos, dft2

FORMA = (64, 80)  # (alto, ancho)


def _imagen_suave(forma, semilla=0):
    """Imagen "natural" sintetica: suma de gaussianas, energia concentrada en bajas frecuencias."""
    rng = np.random.default_rng(semilla)
    alto, ancho = forma
    y, x = np.mgrid[0:alto, 0:ancho]
    img = np.zeros(forma)
    for _ in range(6):
        cy, cx = rng.uniform(0, alto), rng.uniform(0, ancho)
        s = rng.uniform(6, 15)
        img += rng.uniform(0.2, 0.6) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s**2))
    return img / img.max()


def _como_conjunto(picos):
    return {tuple(int(c) for c in p) for p in picos}


# Coseno_2d y agregar_interferencia


@pytest.mark.parametrize("ku, kv", [(5, 3), (20, -7), (0, 9), (13, 0)])
def test_coseno_produce_exactamente_dos_picos_en_mas_menos_u0_v0(ku, kv):
    alto, ancho = FORMA
    A = 0.4
    mag = np.abs(dft2(coseno_2d(FORMA, ku, kv, A)))
    picos = _como_conjunto(np.argwhere(mag > 1e-6 * mag.max()))
    assert picos == {(kv % alto, ku % ancho), (-kv % alto, -ku % ancho)}
    # cada pico lleva la mitad de la energia: A * alto * ancho / 2
    assert mag.max() == pytest.approx(A * alto * ancho / 2)


def test_coseno_tiene_amplitud_y_forma_correctas():
    c = coseno_2d(FORMA, 5, 3, 0.4)
    assert c.shape == FORMA
    assert c[0, 0] == pytest.approx(0.4)  # cos(0) = 1
    assert np.abs(c).max() == pytest.approx(0.4)


def test_agregar_interferencia_suma_sin_recortar():
    img = np.full(FORMA, 0.9)
    r = agregar_interferencia(img, 5, 3, 0.4)
    np.testing.assert_allclose(r - img, coseno_2d(FORMA, 5, 3, 0.4), atol=1e-12)
    assert r.max() > 1  


def test_agregar_interferencia_acepta_enteros():
    img = np.zeros(FORMA, dtype=np.uint8)
    r = agregar_interferencia(img, 5, 3, 0.4)
    assert r.dtype == float
    assert r.min() < 0  


# Detectar picos


def test_detecta_un_coseno_y_su_conjugado():
    E = dft2(coseno_2d(FORMA, 5, 3, 0.4))
    assert _como_conjunto(detectar_picos(E, 0.5, radio_dc=2)) == {(3, 5), (61, 75)}


def test_detecta_dos_interferencias_sobre_una_imagen_suave():
    img = agregar_interferencia(_imagen_suave(FORMA), 20, 7, 0.2)
    img = agregar_interferencia(img, 30, -15, 0.1)
    picos = _como_conjunto(detectar_picos(dft2(img), 0.3))
    alto, ancho = FORMA
    assert picos == {(7, 20), (alto - 7, ancho - 20), (-15 % alto, 30), (15, ancho - 30)}


def test_sin_interferencia_fuera_del_disco_no_hay_picos():
    E = dft2(coseno_2d(FORMA, 5, 3, 0.4))  
    picos = detectar_picos(E, 0.5)
    assert picos.shape == (0, 2)


def test_imagen_constante_no_tiene_picos():
    assert detectar_picos(dft2(np.full(FORMA, 0.5)), 0.5).shape == (0, 2)


@pytest.mark.parametrize("semilla", range(5))
def test_siempre_devuelve_pares_conjugados(semilla):
    rng = np.random.default_rng(semilla)
    img = rng.uniform(0, 1, FORMA)  # espectro con muchos bins parecidos
    alto, ancho = FORMA
    picos = _como_conjunto(detectar_picos(dft2(img), 0.6, radio_dc=1))
    assert picos  # algo encuentra
    assert all(((-v) % alto, (-u) % ancho) in picos for v, u in picos)


def test_nunca_devuelve_el_dc():
    img = agregar_interferencia(_imagen_suave(FORMA), 20, 7, 0.2)
    for radio in (0, 2, 10):
        assert (0, 0) not in _como_conjunto(detectar_picos(dft2(img), 0.01, radio_dc=radio))


def test_formato_de_salida():
    picos = detectar_picos(dft2(coseno_2d(FORMA, 20, 7, 1.0)), 0.5)
    assert picos.ndim == 2 and picos.shape[1] == 2
    assert np.issubdtype(picos.dtype, np.integer)