import numpy as np
from scipy.signal import convolve2d

from psi_lab.lab1.convolucion import (
    convolucion_directa,
    convolucion_fft,
    convolucion_separable_2d,
)


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
