"""C3: notch 2D y PSNR. Propiedades que la teoria garantiza, solo con imagenes sinteticas."""

import numpy as np
import pytest

from psi_lab.lab1.caso_imagen import (
    agregar_interferencia,
    aplicar_mascara,
    detectar_picos,
    dft2,
    notch_gaussiano,
    notch_ideal,
    psnr,
)
from psi_lab.lab1.caso_imagen.notch import pasabajos_gaussiano

FORMA = (64, 80)  # (alto, ancho)
# Interferencias (amplitud, ku, kv); |(kv, ku)| > 10 bins para quedar fuera del DC que excluye
# detectar_picos.
INTERFERENCIAS = ((0.2, 20, 14), (0.15, 30, -18))


def _imagen_suave(forma=FORMA, semilla=0):
    """Imagen \"natural\" sintetica: gaussianas, energia concentrada en bajas frecuencias."""
    rng = np.random.default_rng(semilla)
    alto, ancho = forma
    y, x = np.mgrid[0:alto, 0:ancho]
    img = np.zeros(forma)
    for _ in range(6):
        cy, cx = rng.uniform(0, alto), rng.uniform(0, ancho)
        s = rng.uniform(6, 15)
        img += rng.uniform(0.2, 0.6) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s**2))
    return img / img.max()


def _par(ku, kv, forma=FORMA):
    alto, ancho = forma
    return [(kv % alto, ku % ancho), (-kv % alto, -ku % ancho)]


def _picos_todos():
    return np.array([p for _, ku, kv in INTERFERENCIAS for p in _par(ku, kv)])


def _con_ruido(img):
    for amplitud, ku, kv in INTERFERENCIAS:
        img = agregar_interferencia(img, ku, kv, amplitud)
    return img


# Mascaras


@pytest.mark.parametrize("mascara, param", [(notch_ideal, 2), (notch_gaussiano, 2.0)])
def test_mascara_es_cero_en_cada_pico_y_uno_en_el_dc(mascara, param):
    picos = _picos_todos()
    H = mascara(FORMA, picos, param)
    assert H.shape == FORMA
    assert np.all(H[picos[:, 0], picos[:, 1]] == 0)
    assert H[0, 0] == 1  # nunca se toca el brillo medio


@pytest.mark.parametrize("mascara, param", [(notch_ideal, 3), (notch_gaussiano, 3.0)])
def test_mascara_esta_en_0_1_y_es_simetrica_conjugada(mascara, param):
    H = mascara(FORMA, _picos_todos(), param)
    assert H.min() >= 0 and H.max() <= 1
    # H[v, u] = H[-v, -u] para que la imagen filtrada sea real
    espejo = np.roll(H[::-1, ::-1], shift=(1, 1), axis=(0, 1))
    assert np.allclose(H, espejo, atol=1e-12)


def test_notch_ideal_es_binario_y_su_area_es_la_del_disco():
    H = notch_ideal((101, 101), [(50, 50)], radio=5)
    assert set(np.unique(H)) == {0.0, 1.0}
    n_ceros = int(np.sum(H == 0))
    assert n_ceros == pytest.approx(np.pi * 5**2, rel=0.1)


def test_notch_ideal_usa_distancia_periodica_cerca_del_borde():
    # Un pico en (0, 5) tambien debe borrar los bins del borde opuesto: (alto - 1, 5)
    H = notch_ideal((20, 20), [(0, 5)], radio=1)
    assert H[19, 5] == 0 and H[1, 5] == 0 and H[0, 4] == 0
    assert H[10, 5] == 1


def test_notch_gaussiano_sigma_mayor_borra_mas():
    pico = [(10, 12)]
    chico = notch_gaussiano(FORMA, pico, 1.0)
    grande = notch_gaussiano(FORMA, pico, 4.0)
    assert np.all(grande <= chico + 1e-12)
    assert grande.sum() < chico.sum()


def test_sin_picos_la_mascara_es_identidad():
    sin_picos = np.zeros((0, 2), dtype=int)
    assert np.all(notch_ideal(FORMA, sin_picos, 3) == 1)
    assert np.all(notch_gaussiano(FORMA, sin_picos, 3.0) == 1)


def test_parametros_invalidos_fallan():
    with pytest.raises(ValueError):
        notch_ideal(FORMA, [(1, 1)], -1)
    with pytest.raises(ValueError):
        notch_gaussiano(FORMA, [(1, 1)], 0)
    with pytest.raises(ValueError):
        notch_ideal(FORMA, [1, 2, 3], 1)


# aplicar_mascara


def test_mascara_identidad_devuelve_la_imagen():
    img = _imagen_suave()
    res = aplicar_mascara(img, np.ones(FORMA))
    assert np.allclose(res.real, img) and np.allclose(res.imag, 0)


@pytest.mark.parametrize("mascara, param", [(notch_ideal, 2), (notch_gaussiano, 2.0)])
def test_par_conjugado_completo_da_imagen_real(mascara, param):
    ruidosa = _con_ruido(_imagen_suave())
    res = aplicar_mascara(ruidosa, mascara(FORMA, _picos_todos(), param))
    assert np.allclose(res.imag, 0)


def test_borrar_un_solo_pico_del_par_deja_imagen_compleja():
    _, ku, kv = INTERFERENCIAS[0]
    uno_solo = np.array([_par(ku, kv)[0]])
    res = aplicar_mascara(_con_ruido(_imagen_suave()), notch_ideal(FORMA, uno_solo, 1))
    assert not np.allclose(res.imag, 0, atol=1e-6)


def test_notch_no_toca_el_dc():
    img = _imagen_suave()
    res = aplicar_mascara(_con_ruido(img), notch_ideal(FORMA, _picos_todos(), 3)).real
    assert res.mean() == pytest.approx(img.mean())


def test_aplicar_mascara_exige_la_misma_forma():
    with pytest.raises(ValueError):
        aplicar_mascara(np.zeros((4, 4)), np.ones((4, 5)))


# Flujo: interferencia -> detectar picos -> notch -> PSNR


UMBRAL_PSNR_DB = 35.0  # criterio del grupo: el notch debe dejar el PSNR sobre 35 dB


@pytest.mark.parametrize("mascara, param", [(notch_ideal, 2), (notch_gaussiano, 1.5)])
def test_notch_elimina_la_interferencia_sintetica(mascara, param):
    img = _imagen_suave()
    ruidosa = _con_ruido(img)
    picos = detectar_picos(dft2(ruidosa), umbral=0.3)
    assert len(picos) == 4  # 2 interferencias -> 2 pares conjugados
    limpia = aplicar_mascara(ruidosa, mascara(FORMA, picos, param)).real
    antes, despues = psnr(ruidosa, img), psnr(limpia, img)
    assert despues > UMBRAL_PSNR_DB
    assert despues > antes + 15


def test_notch_sobre_el_dc_oscurece_la_imagen():
    """La trampa 2: si la mascara toca el bin [0, 0] se pierde el brillo medio."""
    img = _imagen_suave()
    res = aplicar_mascara(img, notch_ideal(FORMA, [(0, 0)], 1)).real
    assert res.mean() == pytest.approx(0, abs=1e-12)
    assert psnr(res, img) < 15


# PSNR


def test_psnr_de_imagenes_identicas_es_infinito():
    img = _imagen_suave()
    assert psnr(img, img) == float("inf")


def test_psnr_coincide_con_el_mse_conocido():
    a = np.zeros((10, 10))
    b = a + 0.1  # MSE = 0.01 -> PSNR = 10 log10(1 / 0.01) = 20 dB
    assert psnr(a, b) == pytest.approx(20.0)
    assert psnr(a, b, rango=2.0) == pytest.approx(20.0 + 20 * np.log10(2))


def test_psnr_es_simetrico_y_baja_con_mas_error():
    img = _imagen_suave()
    ruido = np.random.default_rng(1).standard_normal(FORMA)
    chico, grande = img + 0.01 * ruido, img + 0.1 * ruido
    assert psnr(img, chico) == pytest.approx(psnr(chico, img))
    assert psnr(img, chico) > psnr(img, grande)
    # una decada mas de amplitud de ruido = 20 dB menos
    assert psnr(img, chico) - psnr(img, grande) == pytest.approx(20, abs=1e-6)


def test_psnr_exige_la_misma_forma():
    with pytest.raises(ValueError):
        psnr(np.zeros((3, 3)), np.zeros((3, 4)))


# Pasabajos gaussiano (linea base de comparacion)


def test_pasabajos_gaussiano_conserva_forma_y_constantes():
    plana = np.full(FORMA, 0.7)
    res = pasabajos_gaussiano(plana, 2.0)
    assert res.shape == FORMA
    assert np.allclose(res, 0.7)  # ganancia 1 en DC, sin oscurecer los bordes


def test_pasabajos_gaussiano_atenua_un_coseno_segun_la_teoria():
    # La transformada de una gaussiana es otra gaussiana: |H(f)| = exp(-2 pi^2 sigma^2 f^2)
    alto, ancho = 128, 128
    ku, sigma = 12, 2.0
    x = np.arange(ancho)[None, :] * np.ones((alto, 1))
    img = np.cos(2 * np.pi * ku * x / ancho)
    res = pasabajos_gaussiano(img, sigma)
    centro = res[:, 32:-32]  # lejos de los bordes, donde el reflejo no importa
    ganancia = np.abs(centro).max() / np.abs(img[:, 32:-32]).max()
    esperada = np.exp(-2 * np.pi**2 * sigma**2 * (ku / ancho) ** 2)
    assert ganancia == pytest.approx(esperada, rel=0.02)


def test_pasabajos_gaussiano_coincide_con_scipy():
    from scipy.ndimage import gaussian_filter  # referencia, solo dentro de las pruebas

    img = _imagen_suave()
    esperado = gaussian_filter(img, sigma=2.0, mode="mirror", truncate=3.0)
    assert np.allclose(pasabajos_gaussiano(img, 2.0), esperado, atol=1e-3)


def test_pasabajos_gaussiano_sigma_invalido_falla():
    with pytest.raises(ValueError):
        pasabajos_gaussiano(np.zeros((4, 4)), 0)
