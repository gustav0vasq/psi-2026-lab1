"""Convolucion lineal vs circular """

import time

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.convolucion import convolucion_directa, convolucion_fft
from psi_lab.lab1.dft import dft, idft
from psi_lab.lab1.figuras import figura


@figura("convolucion_circular")
def _(salida):
    # pulso cerca del final para que la cola se pase del largo de x
    x = np.zeros(40)
    x[25:] = 1.0
    # h: exponencial que decae, largo 15
    h = np.exp(-np.arange(15) / 4)

    lineal = convolucion_fft(x, h)

    # sin relleno, yodo queda de largo len(x), h se completa solo hasta ahi
    h_rell = np.concatenate([h, np.zeros(len(x) - len(h))])
    circular = idft(dft(x) * dft(h_rell)).real

    fig = Figure(figsize=(8, 5))
    ax_lin, ax_circ = fig.subplots(2, 1, sharex=True)

    ax_lin.stem(np.arange(len(lineal)), lineal)
    ax_lin.axvline(len(x) - 0.5, ls="--", color="gray")
    ax_lin.set(ylabel="y[n]", title=f"Lineal, con relleno de ceros (largo {len(lineal)})")

    ax_circ.stem(np.arange(len(circular)), circular)
    ax_circ.axvspan(-0.5, len(h) - 1.5, alpha=0.2, color="red")
    ax_circ.set(xlabel="n", ylabel="y[n]",
                title=f"Circular, sin relleno (largo {len(circular)})")

    fig.tight_layout()
    fig.savefig(salida / "convolucion_circular.png", dpi=120)


@figura("convolucion_tiempos")
def _(salida):
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    largos = [4, 8, 16, 32, 64, 128, 256, 512]
    t_dir = []
    t_fft = []
    for m in largos:
        h = rng.normal(size=m)
        mejor_dir = np.inf
        mejor_fft = np.inf
        for _ in range(3):
            inicio = time.perf_counter()
            convolucion_directa(x, h)
            mejor_dir = min(mejor_dir, time.perf_counter() - inicio)
            inicio = time.perf_counter()
            convolucion_fft(x, h)
            mejor_fft = min(mejor_fft, time.perf_counter() - inicio)
        t_dir.append(mejor_dir)
        t_fft.append(mejor_fft)

    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.loglog(largos, t_dir, "o-", label="directa")
    ax.loglog(largos, t_fft, "s-", label="por DFT")
    ax.set(xlabel="largo de h", ylabel="tiempo [s]",
           title=f"Tiempo de convolucion, len(x) = {len(x)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "convolucion_tiempos.png", dpi=120)
