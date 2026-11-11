"""Pruebas de dft, idft y espectro: propiedades teoricas (np.fft solo como referencia)."""

import numpy as np
import pytest

from psi_lab.lab1.dft import _ventana, dft, espectro, idft

LARGOS = [1, 2, 7, 16, 63, 128, 500]


def _aleatoria(N, compleja=False, semilla=0):
    rng = np.random.default_rng(semilla)
    x = rng.standard_normal(N)
    if compleja:
        x = x + 1j * rng.standard_normal(N)
    return x


# dft e idft


@pytest.mark.parametrize("N", LARGOS)
@pytest.mark.parametrize("compleja", [False, True])
def test_dft_coincide_con_np_fft(N, compleja):
    x = _aleatoria(N, compleja)
    np.testing.assert_allclose(dft(x), np.fft.fft(x), atol=1e-9 * N)


@pytest.mark.parametrize("N", LARGOS)
def test_idft_recupera_x(N):
    x = _aleatoria(N, compleja=True)
    np.testing.assert_allclose(idft(dft(x)), x, atol=1e-10 * N)


@pytest.mark.parametrize("N", [8, 9, 100])
def test_simetria_hermitica_para_x_real(N):
    X = dft(_aleatoria(N))
    k = np.arange(1, N)
    np.testing.assert_allclose(X[N - k], np.conj(X[k]), atol=1e-9 * N)


@pytest.mark.parametrize("N", LARGOS)
def test_parseval(N):
    x = _aleatoria(N, compleja=True)
    X = dft(x)
    assert np.sum(np.abs(x) ** 2) == pytest.approx(np.sum(np.abs(X) ** 2) / N)


def test_impulso_da_espectro_plano():
    x = np.zeros(32)
    x[0] = 1
    np.testing.assert_allclose(dft(x), np.ones(32), atol=1e-12)


# espectro 


@pytest.mark.parametrize("ventana", ["rect", "hann", "hamming"])
@pytest.mark.parametrize("f, A", [(50.0, 1.0), (123.0, 0.3), (10.0, 2.0)])
def test_tono_da_pico_en_f_con_altura_A(ventana, f, A):
    fs, N = 1000, 1000  # bins cada fs/N = 1 Hz: los tonos caen justo en un bin
    t = np.arange(N) / fs
    freqs, amp = espectro(A * np.cos(2 * np.pi * f * t), fs, ventana=ventana)
    assert freqs[np.argmax(amp)] == pytest.approx(f)
    # Hann y Hamming simetricas (N - 1) fugan ~1e-7 incluso en un bin: tolerancia mayor.
    assert amp.max() == pytest.approx(A, rel=1e-9 if ventana == "rect" else 1e-5)


@pytest.mark.parametrize("N, nfft", [(100, None), (100, 800), (99, None), (99, 256)])
def test_largo_y_eje_de_frecuencias(N, nfft):
    fs = 500.0
    freqs, amp = espectro(np.ones(N), fs, nfft=nfft)
    M = N if nfft is None else nfft
    assert freqs.shape == amp.shape == (M // 2 + 1,)
    assert freqs[0] == 0
    np.testing.assert_allclose(np.diff(freqs), fs / M)
    assert freqs[-1] <= fs / 2


def test_constante_da_solo_dc():
    _, amp = espectro(np.full(64, 0.7), fs=64)
    assert amp[0] == pytest.approx(0.7)
    np.testing.assert_allclose(amp[1:], 0, atol=1e-12)


def test_nyquist_no_se_duplica():
    x = np.cos(np.pi * np.arange(8))  # tono en fs/2: +1, -1, +1, ...
    freqs, amp = espectro(x, fs=8)
    assert freqs[-1] == 4
    assert amp[-1] == pytest.approx(1.0)


def test_relleno_con_ceros_conserva_el_pico():
    fs, N = 1000, 250  # bins cada 4 Hz; 52 Hz cae justo en uno
    x = np.cos(2 * np.pi * 52 * np.arange(N) / fs)
    f1, a1 = espectro(x, fs)
    f8, a8 = espectro(x, fs, nfft=8 * N)
    assert f8[np.argmax(a8)] == pytest.approx(f1[np.argmax(a1)])
    assert a8.max() == pytest.approx(a1.max())
    np.testing.assert_allclose(a8[::8], a1, atol=1e-9)  # los bins originales se conservan


@pytest.mark.parametrize("nombre, referencia", [("hann", np.hanning), ("hamming", np.hamming)])
def test_ventanas_coinciden_con_numpy(nombre, referencia):
    np.testing.assert_allclose(_ventana(nombre, 51), referencia(51), atol=1e-12)


def test_hann_reduce_la_fuga():
    fs, N = 1000, 1000
    x = np.cos(2 * np.pi * 50.5 * np.arange(N) / fs)  # medio bin: fuga maxima con rect
    freqs, rect = espectro(x, fs, "rect")
    _, hann = espectro(x, fs, "hann")
    lejos = np.abs(freqs - 50.5) > 20
    assert hann[lejos].max() < rect[lejos].max() / 100


@pytest.mark.parametrize("kwargs", [{"ventana": "kaiser"}, {"nfft": 10}])
def test_espectro_rechaza_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        espectro(np.ones(32), fs=100, **kwargs)