import numpy as np
import pytest

from psi_lab.lab1.caso_imagen import dft2, espectro_log, fase, idft2


@pytest.fixture
def img():
    return np.random.default_rng(0).random((12, 20))  # no cuadrada: detecta ejes cambiados


def test_dft2_coincide_con_fft2(img):
    assert np.allclose(dft2(img), np.fft.fft2(img))


def test_idft2_recupera_la_imagen(img):
    assert np.allclose(idft2(dft2(img)).real, img)
    assert np.allclose(idft2(dft2(img)).imag, 0)


def test_idft2_coincide_con_ifft2():
    E = np.random.default_rng(1).standard_normal((8, 6)) + 0j
    assert np.allclose(idft2(E), np.fft.ifft2(E))


def test_dc_es_la_suma_de_los_pixeles(img):
    assert dft2(img)[0, 0] == pytest.approx(img.sum())


def test_simetria_conjugada_de_imagen_real(img):
    E = dft2(img)
    alto, ancho = img.shape
    v, u = np.meshgrid(np.arange(alto), np.arange(ancho), indexing="ij")
    assert np.allclose(E[(-v) % alto, (-u) % ancho], np.conj(E))


def test_parseval_2d(img):
    E = dft2(img)
    assert np.sum(np.abs(E) ** 2) / img.size == pytest.approx(np.sum(img**2))


@pytest.mark.parametrize("ku,kv", [(3, 0), (0, 2), (4, 3), (7, 5)])
def test_coseno_2d_da_exactamente_dos_picos(ku, kv):
    alto, ancho, A = 16, 24, 2.0
    y, x = np.mgrid[:alto, :ancho]
    E = dft2(A * np.cos(2 * np.pi * (ku * x / ancho + kv * y / alto)))
    picos = np.argwhere(np.abs(E) > 1e-8)
    assert {tuple(p) for p in picos} == {(kv, ku), ((alto - kv) % alto, (ancho - ku) % ancho)}
    assert np.abs(E[kv, ku]) == pytest.approx(A * alto * ancho / 2)


def test_espectro_log_centra_el_dc():
    E = dft2(np.full((10, 14), 0.5))
    e = espectro_log(E)
    assert e.shape == (10, 14)
    assert np.unravel_index(np.argmax(e), e.shape) == (5, 7)


def test_espectro_log_no_modifica_el_arreglo_de_trabajo(img):
    E = dft2(img)
    copia = E.copy()
    espectro_log(E)
    assert np.array_equal(E, copia)


def test_fase_de_un_impulso_desplazado():
    # Un impulso en (a, b) tiene fase -2 pi (v a / alto + u b / ancho).
    alto, ancho, a, b = 8, 10, 2, 3
    img = np.zeros((alto, ancho))
    img[a, b] = 1
    v, u = np.meshgrid(np.arange(alto), np.arange(ancho), indexing="ij")
    esperada = np.exp(-2j * np.pi * (v * a / alto + u * b / ancho))
    f = fase(dft2(img))
    assert np.all(np.abs(f) <= np.pi)
    assert np.allclose(np.exp(1j * f), esperada)


def test_exige_arreglos_2d():
    for f in (dft2, idft2, espectro_log, fase):
        with pytest.raises(ValueError):
            f(np.zeros(8))
