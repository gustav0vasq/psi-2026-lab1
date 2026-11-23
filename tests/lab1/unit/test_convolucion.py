import numpy as np
import pytest
from scipy.signal import convolve2d

from psi_lab.lab1.convolucion import (
    convolucion_directa,
    convolucion_fft,
    convolucion_separable_2d,
    fft_radix2,
    ifft_radix2,
)
from psi_lab.lab1.dft import dft


def test_directa_coincide_np_convolve():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    h = rng.normal(size=7)
    assert np.allclose(convolucion_directa(x, h), np.convolve(x, h))


def test_impulso_desplazado_desplaza_senal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    h = np.zeros(4)
    h[3] = 1 # Genera un impulso en la posicion 3
    assert np.allclose(convolucion_directa(x,h), np.concatenate([np.zeros(3), x]))
    

def test_conmutativa():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    h = rng.normal(size=7)
    assert np.allclose(convolucion_directa(x,h), convolucion_directa(h,x))


def test_fft_coincide_np_convolve():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    h = rng.normal(size=7)
    assert np.allclose(convolucion_fft(x, h), np.convolve(x, h))


def test_fft_coincide_con_directa():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    h = rng.normal(size=7)
    assert np.allclose(convolucion_fft(x, h), convolucion_directa(x, h))


def test_separable_coincide_con_convolve2d():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(20, 30))
    h_filas = rng.normal(size=5)
    h_cols = rng.normal(size=3)
    
    nucleo = np.outer(h_cols, h_filas)
    esperado = convolve2d(img, nucleo)
    assert np.allclose(convolucion_separable_2d(img, h_filas, h_cols), esperado)


# --- FFT radix-2 propia -------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 2, 4, 64, 1024])
def test_fft_radix2_coincide_con_np_fft_y_con_dft(N):
    x = np.random.default_rng(N).normal(size=N) + 1j * np.random.default_rng(N + 1).normal(size=N)
    assert np.allclose(fft_radix2(x), np.fft.fft(x))
    assert np.allclose(fft_radix2(x), dft(x))


@pytest.mark.parametrize("N", [2, 16, 256])
def test_ifft_radix2_recupera_x(N):
    x = np.random.default_rng(N).normal(size=N)
    assert np.allclose(ifft_radix2(fft_radix2(x)).real, x)
    assert np.allclose(ifft_radix2(fft_radix2(x)).imag, 0)


def test_fft_radix2_exige_potencia_de_2():
    with pytest.raises(ValueError):
        fft_radix2(np.zeros(12))


@pytest.mark.parametrize("n_x,n_h", [(1, 1), (5, 1), (1, 9), (64, 64), (100, 33), (37, 128)])
def test_fft_coincide_np_convolve_con_largos_cualquiera(n_x, n_h):
    rng = np.random.default_rng(n_x * 1000 + n_h)
    x, h = rng.normal(size=n_x), rng.normal(size=n_h)
    y = convolucion_fft(x, h)
    assert y.size == n_x + n_h - 1
    assert np.allclose(y, np.convolve(x, h))
