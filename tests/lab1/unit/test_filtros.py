"""Tarea 4: filtros FIR por ventana. Cada prueba verifica una propiedad teorica.

np.fft se usa solo aqui, como referencia para validar respuesta_frecuencia.
"""

import numpy as np
import pytest

from psi_lab.lab1.filtros import (
    VENTANAS,
    crear_ventana,
    filtrar,
    fir_pasaaltos,
    fir_pasabajos,
    fir_pasabanda,
    fir_rechazabanda,
    medir_pasabajos,
    respuesta_frecuencia,
)
from psi_lab.lab1.sinteticos import suma_tonos, tono

FS = 1000
N = 101


def _ganancia(h, f, fs=FS):
    """|H(f)| evaluada por definicion en una sola frecuencia."""
    k = np.arange(len(h))
    return abs(np.sum(h * np.exp(-2j * np.pi * f * k / fs)))


# --- ventanas -------------------------------------------------------------------------


@pytest.mark.parametrize("nombre", VENTANAS)
def test_ventana_simetrica_y_maximo_al_centro(nombre):
    w = crear_ventana(nombre, 51)
    assert np.allclose(w, w[::-1])
    assert w[25] == pytest.approx(1.0)
    assert np.all(w <= 1 + 1e-12)


def test_ventana_desconocida_falla():
    with pytest.raises(ValueError):
        crear_ventana("kaiser", 11)


# --- pasabajos ------------------------------------------------------------------------


@pytest.mark.parametrize("nombre", VENTANAS)
def test_pasabajos_ganancia_uno_en_dc_y_fase_lineal(nombre):
    h = fir_pasabajos(100, FS, N, nombre)
    assert h.size == N
    assert h.sum() == pytest.approx(1.0)
    assert np.allclose(h, h[::-1])  # simetria -> fase lineal


@pytest.mark.parametrize("nombre", ["hann", "hamming", "blackman"])
def test_pasabajos_media_amplitud_en_el_corte(nombre):
    # La respuesta de la sinc enventanada es simetrica en torno a fc: |H(fc)| = 1/2 (-6 dB)
    h = fir_pasabajos(100, FS, N, nombre)
    assert _ganancia(h, 100) == pytest.approx(0.5, abs=0.01)


def test_pasabajos_fase_lineal_retardo_de_grupo():
    # H(f) = A(f) exp(-j 2 pi f M / fs), con A real y M = (N-1)/2
    h = fir_pasabajos(100, FS, N)
    freqs, H = respuesta_frecuencia(h, FS)
    A = H * np.exp(2j * np.pi * freqs * (N - 1) / 2 / FS)
    assert np.allclose(A.imag, 0, atol=1e-12)


@pytest.mark.parametrize("fc", [0, 500, 600, -10])
def test_corte_fuera_de_rango_falla(fc):
    with pytest.raises(ValueError):
        fir_pasabajos(fc, FS, N)


# --- derivados del pasabajos ----------------------------------------------------------


def test_pasaaltos_es_el_complemento_del_pasabajos():
    hl, hh = fir_pasabajos(100, FS, N), fir_pasaaltos(100, FS, N)
    delta = np.zeros(N)
    delta[N // 2] = 1
    assert np.allclose(hl + hh, delta)
    assert hh.sum() == pytest.approx(0, abs=1e-12)  # ganancia 0 en DC
    assert _ganancia(hh, FS / 2) == pytest.approx(1, abs=0.01)


def test_pasabanda_pasa_el_centro_y_rechaza_los_extremos():
    h = fir_pasabanda(100, 200, FS, N)
    assert h.sum() == pytest.approx(0, abs=1e-12)
    assert _ganancia(h, 150) == pytest.approx(1, abs=0.01)
    assert _ganancia(h, 0) < 1e-3 and _ganancia(h, FS / 2) < 1e-2


def test_rechazabanda_es_el_complemento_del_pasabanda():
    hb, hr = fir_pasabanda(100, 200, FS, N), fir_rechazabanda(100, 200, FS, N)
    assert np.allclose(hb + hr, fir_pasabajos(100, FS, N) + fir_pasaaltos(100, FS, N))
    assert hr.sum() == pytest.approx(1)
    assert _ganancia(hr, 150) < 0.01
    assert _ganancia(hr, FS / 2) == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize("fun", [fir_pasaaltos, fir_rechazabanda])
def test_n_coef_par_falla_donde_se_necesita_la_delta(fun):
    args = (100, FS, 100) if fun is fir_pasaaltos else (100, 200, FS, 100)
    with pytest.raises(ValueError):
        fun(*args)


def test_pasabanda_con_bordes_invertidos_falla():
    with pytest.raises(ValueError):
        fir_pasabanda(200, 100, FS, N)


# --- respuesta en frecuencia ----------------------------------------------------------


def test_respuesta_frecuencia_coincide_con_la_fft_rellenada():
    h = fir_pasabanda(50, 120, FS, 61, "blackman")
    freqs, H = respuesta_frecuencia(h, FS, n=512)
    assert freqs[0] == 0 and freqs[-1] < FS / 2
    assert np.allclose(freqs, np.fft.rfftfreq(1024, 1 / FS)[:512])
    assert np.allclose(H, np.fft.fft(h, 1024)[:512])


# --- filtrar --------------------------------------------------------------------------


def test_filtrar_mantiene_el_largo():
    assert filtrar(np.ones(300), fir_pasabajos(100, FS, N)).size == 300


def test_filtrar_banda_de_paso_sin_desfase():
    # Un tono en la banda de paso sale igual (amplitud y fase), fuera de los bordes
    x = tono(20, FS, 1)
    y = filtrar(x, fir_pasabajos(100, FS, N))
    centro = slice(N, -N)
    assert np.allclose(y[centro], x[centro], atol=5e-3)


def test_filtrar_separa_dos_tonos():
    # 50 Hz en la banda de paso, 300 Hz en la de rechazo (atenuacion Hamming ~53 dB)
    x = suma_tonos([50, 300], FS, 1)
    y = filtrar(x, fir_pasabajos(150, FS, N))
    centro = slice(N, -N)
    resto = y[centro] - tono(50, FS, 1)[centro]
    assert np.max(np.abs(resto)) < 10 ** (-50 / 20) + 5e-3


# --- caracterizacion: ancho de transicion y atenuacion --------------------------------

# Valores teoricos del metodo de la ventana (Oppenheim & Schafer, tabla 7.2):
# ancho de transicion ~ C fs / N  y atenuacion minima en la banda de rechazo
TEORIA = {"rect": (0.9, 21), "hann": (3.1, 44), "hamming": (3.3, 53), "blackman": (5.5, 74)}


@pytest.mark.parametrize("nombre", VENTANAS)
@pytest.mark.parametrize("n_coef", [51, 101, 201])
def test_medicion_coincide_con_la_tabla_teorica(nombre, n_coef):
    c, atenuacion = TEORIA[nombre]
    ancho, a = medir_pasabajos(fir_pasabajos(100, FS, n_coef, nombre), FS, 100)
    assert ancho * n_coef / FS == pytest.approx(c, rel=0.15)
    assert a == pytest.approx(atenuacion, abs=3)


@pytest.mark.parametrize("nombre", VENTANAS)
def test_atenuacion_no_depende_de_n_coef_y_el_ancho_si(nombre):
    (d1, a1), (d2, a2) = (
        medir_pasabajos(fir_pasabajos(100, FS, n, nombre), FS, 100) for n in (101, 201)
    )
    assert a1 == pytest.approx(a2, abs=1.5)  # la ventana fija la atenuacion
    assert d1 / d2 == pytest.approx(2, rel=0.15)  # el ancho es ~ 1 / N


def test_ventanas_ordenadas_por_atenuacion_y_por_ancho():
    medidas = [medir_pasabajos(fir_pasabajos(100, FS, N, v), FS, 100) for v in VENTANAS]
    anchos, atenuaciones = zip(*medidas)
    assert list(atenuaciones) == sorted(atenuaciones)
    assert list(anchos) == sorted(anchos)


# --- pruebas que pide explicitamente el enunciado (Tarea 4.4) -------------------------


@pytest.mark.parametrize("n_coef", [51, 101, 201])
def test_hamming_ganancia_1_en_paso_y_bajo_menos_40_db_en_rechazo(n_coef):
    fc = 100
    h = fir_pasabajos(fc, FS, n_coef, "hamming")
    freqs, H = respuesta_frecuencia(h, FS)
    db = 20 * np.log10(np.abs(H))
    ancho = 3.3 * FS / n_coef  # ancho de transicion teorico de Hamming
    assert np.allclose(np.abs(H[freqs <= fc - ancho / 2]), 1, atol=0.01)
    assert db[freqs >= fc + ancho / 2].max() < -40


@pytest.mark.parametrize("posicion", [100, 250, 400])
def test_filtrar_no_desplaza_un_pulso(posicion):
    # Un pulso gaussiano ancho (de banda angosta) sale con su maximo en el mismo instante
    n = np.arange(500)
    pulso = np.exp(-(((n - posicion) / 10) ** 2) / 2)
    y = filtrar(pulso, fir_pasabajos(100, FS, N))
    assert np.argmax(y) == posicion
    # el pulso cabe en la banda de paso: solo lo deforma el rizado de Hamming (~0.002)
    assert np.allclose(y, pulso, atol=3e-3)


def test_filtrar_impulso_devuelve_h_centrada():
    # La respuesta a un impulso en n0 es h, con su centro en n0 (retardo compensado)
    h = fir_pasabajos(100, FS, N)
    x = np.zeros(300)
    x[150] = 1
    y = filtrar(x, h)
    assert np.argmax(y) == 150
    assert np.allclose(y[150 - N // 2 : 150 + N // 2 + 1], h)
