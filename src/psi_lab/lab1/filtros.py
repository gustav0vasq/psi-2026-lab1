"""Parte comun 4: filtros FIR por el metodo de la ventana.

Solo NumPy (sin np.fft ni scipy.signal). Todos los filtros se derivan del pasabajos:

- pasabajos:     h_lp = sinc ideal truncada * ventana, normalizada a ganancia 1 en DC
- pasaaltos:     delta[n - M] - h_lp           (inversion espectral, M = (n_coef - 1) / 2)
- pasabanda:     h_lp(f2) - h_lp(f1)
- rechazabanda:  delta[n - M] - h_pasabanda(f1, f2)

Los coeficientes son simetricos (fase lineal, retardo de grupo M muestras). Pasaaltos y
rechazabanda exigen n_coef impar: con n_coef par la respuesta vale 0 en fs/2 por construccion
(FIR tipo II) y el retardo M no es entero, asi que la delta no existe.
"""

import numpy as np

VENTANAS = ("rect", "hann", "hamming", "blackman")


def crear_ventana(nombre, n_coef):
    """Ventana simetrica de n_coef muestras: 'rect', 'hann', 'hamming' o 'blackman'."""
    if n_coef < 1:
        raise ValueError("n_coef debe ser >= 1")
    if n_coef == 1:
        return np.ones(1)
    n = np.arange(n_coef)
    a = 2 * np.pi * n / (n_coef - 1)
    if nombre == "rect":
        return np.ones(n_coef)
    if nombre == "hann":
        return 0.5 - 0.5 * np.cos(a)
    if nombre == "hamming":
        return 0.54 - 0.46 * np.cos(a)
    if nombre == "blackman":
        return 0.42 - 0.5 * np.cos(a) + 0.08 * np.cos(2 * a)
    raise ValueError(f"ventana desconocida: {nombre!r}; opciones: {VENTANAS}")


def _validar_corte(f, fs):
    if not 0 < f < fs / 2:
        raise ValueError(f"la frecuencia de corte {f} Hz debe estar en (0, fs/2 = {fs / 2}) Hz")


def _delta(n_coef):
    if n_coef % 2 == 0:
        raise ValueError("pasaaltos y rechazabanda requieren n_coef impar (FIR tipo I)")
    d = np.zeros(n_coef)
    d[(n_coef - 1) // 2] = 1.0
    return d


def fir_pasabajos(fc, fs, n_coef, ventana="hamming"):
    """Sinc truncada y enventanada, normalizada a ganancia 1 en DC.

    fc: corte [Hz] en (0, fs/2); fs [Hz]; n_coef: numero de coeficientes.
    Retorna h, arreglo 1D de largo n_coef, simetrico, con sum(h) = 1.
    """
    _validar_corte(fc, fs)
    w = crear_ventana(ventana, n_coef)
    m = np.arange(n_coef) - (n_coef - 1) / 2
    fn = fc / fs  # corte normalizado [ciclos/muestra]
    h = 2 * fn * np.sinc(2 * fn * m) * w  # np.sinc(x) = sin(pi x) / (pi x)
    return h / h.sum()


def fir_pasaaltos(fc, fs, n_coef, ventana="hamming"):
    """Pasaaltos por inversion espectral: delta - pasabajos. n_coef impar. Ganancia 0 en DC."""
    return _delta(n_coef) - fir_pasabajos(fc, fs, n_coef, ventana)


def fir_pasabanda(f1, f2, fs, n_coef, ventana="hamming"):
    """Pasabanda entre f1 < f2 [Hz]: diferencia de dos pasabajos. Ganancia 0 en DC."""
    if not f1 < f2:
        raise ValueError("se requiere f1 < f2")
    return fir_pasabajos(f2, fs, n_coef, ventana) - fir_pasabajos(f1, fs, n_coef, ventana)


def fir_rechazabanda(f1, f2, fs, n_coef, ventana="hamming"):
    """Rechazabanda entre f1 < f2 [Hz]: delta - pasabanda. n_coef impar. Ganancia 1 en DC."""
    return _delta(n_coef) - fir_pasabanda(f1, f2, fs, n_coef, ventana)


def respuesta_frecuencia(h, fs, n=4096):
    """Retorna (freqs, H) con H complejo.

    H(f) = sum_k h[k] exp(-2j pi f k / fs), evaluada por definicion (producto matricial) en
    n frecuencias freqs = i fs / (2 n), i = 0..n-1, es decir en [0, fs/2). Es la misma
    grilla que la mitad de una DFT de 2n puntos de h rellenado con ceros.
    """
    h = np.asarray(h, dtype=float)
    freqs = np.arange(n) * fs / (2 * n)
    k = np.arange(h.size)
    H = np.exp(-2j * np.pi * np.outer(freqs / fs, k)) @ h
    return freqs, H


def filtrar(x, h):
    """Aplica h a x (largo igual a x), compensando el retardo (n_coef - 1) / 2.

    Convolucion lineal completa y recorte de la parte central, asi una componente en la banda
    de paso sale alineada con la entrada (sin desfase). Para n_coef par el retardo se redondea
    hacia abajo. Los extremos (n_coef // 2 muestras) sufren el transitorio del relleno con ceros.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.convolve(x, h)  # convolucion lineal de NumPy (no es FFT)
    retardo = (h.size - 1) // 2
    return y[retardo : retardo + x.size]


def medir_pasabajos(h, fs, fc, n=1 << 15):
    """Caracteriza un pasabajos: retorna (ancho_transicion [Hz], atenuacion [dB]).

    Definicion (la clasica de los filtros por ventana, donde el rizado de la banda de paso y
    el de la de rechazo son ~iguales, delta):
      1. f_nulo: primer minimo local de |H| por sobre fc (fin del lobulo principal).
      2. delta = max |H(f)| para f >= f_nulo; atenuacion = -20 log10(delta).
      3. f_p = mayor f < fc con |H| >= 1 - delta; f_s = menor f > fc con |H| <= delta.
      4. ancho de transicion = f_s - f_p.
    """
    freqs, H = respuesta_frecuencia(h, fs, n)
    mag = np.abs(H)
    arriba = np.nonzero(freqs > fc)[0]
    d = np.diff(mag[arriba])
    minimos = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    if minimos.size == 0:
        raise ValueError("no se encontro el fin del lobulo principal; aumente n_coef")
    i_nulo = arriba[minimos[0] + 1]
    delta = mag[i_nulo:].max()
    f_p = freqs[(freqs < fc) & (mag >= 1 - delta)].max()
    f_s = freqs[(freqs > fc) & (mag <= delta)].min()
    return f_s - f_p, -20 * np.log10(delta)
