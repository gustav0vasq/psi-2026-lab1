"""Figuras y tablas de la Tarea 2: DFT, resolucion y fuga espectral. Dueno: R2."""

import time

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.dft import dft
from psi_lab.lab1.figuras import figura, guardar_tabla

N_TIEMPOS = 2 ** np.arange(4, 13) 


def _medir(fun, x, repeticiones):
    """Mejor tiempo [s] de `repeticiones` llamadas a fun(x): el minimo filtra interrupciones."""
    mejor = np.inf
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        fun(x)
        mejor = min(mejor, time.perf_counter() - inicio)
    return mejor


def _pendiente(N, t, desde):
    """Pendiente de log10(t) vs log10(N) usando solo N >= desde."""
    usar = N >= desde
    return np.polyfit(np.log10(N[usar]), np.log10(t[usar]), 1)[0]


@figura("dft_tiempos")
def _(salida):
    """Tiempo de dft (O(N^2)) vs np.fft.fft (O(N log N)) en escala log-log."""
    rng = np.random.default_rng(0)
    t_dft, t_fft = [], []
    for N in N_TIEMPOS:
        x = rng.standard_normal(N)
        t_dft.append(_medir(dft, x, repeticiones=3 if N >= 2048 else 10))
        t_fft.append(_medir(np.fft.fft, x, repeticiones=200))
    t_dft, t_fft = np.array(t_dft), np.array(t_fft)

    # Para N chico domina el costo fijo de llamar a Python/NumPy, no el algoritmo:
    # la pendiente se ajusta solo en N grande.
    desde = 256
    p_dft = _pendiente(N_TIEMPOS, t_dft, desde)
    p_fft = _pendiente(N_TIEMPOS, t_fft, desde)

    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ax.loglog(N_TIEMPOS, t_dft, "o-", label=f"dft (pendiente {p_dft:.2f})")
    ax.loglog(N_TIEMPOS, t_fft, "s-", label=f"np.fft.fft (pendiente {p_fft:.2f})")
    # Referencias O(N^2) y O(N log N), ancladas al ultimo punto medido
    Nf = N_TIEMPOS.astype(float)
    ax.loglog(Nf, t_dft[-1] * (Nf / Nf[-1]) ** 2, "k--", lw=0.8, label="referencia N²")
    ref = Nf * np.log2(Nf)
    ax.loglog(Nf, t_fft[-1] * ref / ref[-1], "k:", lw=0.8, label="referencia N log N")
    ax.axvspan(N_TIEMPOS[0], desde, color="0.9", zorder=0)
    ax.set(xlabel="N [muestras]", ylabel="tiempo [s]",
           title="DFT por definicion vs. FFT (zona gris: fuera del ajuste)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "dft_tiempos.png", dpi=120)

    guardar_tabla(
        salida,
        "dft_tiempos",
        ["N", "t_dft_s", "t_fft_s", "razon_dft_fft"],
        [[int(N), f"{a:.3e}", f"{b:.3e}", f"{a / b:.0f}"] for N, a, b in zip(N_TIEMPOS, t_dft, t_fft)],
    )