import numpy as np
import pytest

from psi_lab.lab1.muestreo import frecuencia_alias, muestrear
from psi_lab.lab1.sinteticos import tono


def test_muestrear_largo_y_paso():
    t, x = muestrear(lambda t: np.sin(2 * np.pi * 5 * t), fs=100, duracion=2)
    assert t.size == x.size == 200
    assert t[1] - t[0] == pytest.approx(1 / 100)

# --- frecuencia_alias ---------------------------------------------------------------------


def test_frecuencia_alias_sin_aliasing_bajo_nyquist():
    # Por debajo de fs/2 la frecuencia aparente es la propia.
    assert frecuencia_alias(3, 10) == pytest.approx(3)
    assert frecuencia_alias(5, 10) == pytest.approx(5)  # justo en Nyquist


def test_frecuencia_alias_de_las_frecuencias_de_la_tarea():
    # f = 3, 7, 13, 23 con fs = 10: 7 = 10 - 3, 13 = 10 + 3, 23 = 2 * 10 + 3.
    for f in (3, 7, 13, 23):
        assert frecuencia_alias(f, 10) == pytest.approx(3)


def test_frecuencia_alias_multiplo_de_fs_es_continua():
    assert frecuencia_alias(30, 10) == pytest.approx(0)


@pytest.mark.parametrize("semilla", range(20))
def test_frecuencia_alias_invariancias(semilla):
    rng = np.random.default_rng(semilla)
    fs, f = rng.uniform(5, 500), rng.uniform(0, 2000)
    a = frecuencia_alias(f, fs)
    assert 0 <= a <= fs / 2 + 1e-9
    assert frecuencia_alias(f + 3 * fs, fs) == pytest.approx(a)  # periodica en fs
    assert frecuencia_alias(-f, fs) == pytest.approx(a)  # par
    assert frecuencia_alias(a, fs) == pytest.approx(a)  # idempotente


@pytest.mark.parametrize("semilla", range(20))
def test_frecuencia_alias_produce_las_mismas_muestras(semilla):
    # Definicion de alias: f y su frecuencia aparente generan exactamente las mismas muestras.
    rng = np.random.default_rng(semilla)
    fs, f = rng.uniform(5, 500), rng.uniform(0, 2000)
    _, x = muestrear(lambda t: np.cos(2 * np.pi * f * t), fs, 1)
    _, y = muestrear(lambda t: np.cos(2 * np.pi * frecuencia_alias(f, fs) * t), fs, 1)
    assert np.allclose(x, y, atol=1e-6)


@pytest.mark.parametrize("semilla", range(30))
def test_pico_del_espectro_cae_en_la_frecuencia_alias(semilla):
    # Con duracion 1 s y f entera, el tono cae exacto en un bin de la FFT (sin fuga).
    rng = np.random.default_rng(semilla)
    fs = int(rng.integers(20, 400))
    f = int(rng.integers(1, 5 * fs))
    x = tono(f, fs, 1)
    pico = np.fft.rfftfreq(x.size, 1 / fs)[np.argmax(np.abs(np.fft.rfft(x)))]
    assert pico == pytest.approx(frecuencia_alias(f, fs))
