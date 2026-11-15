"""Convolucion lineal vs circular """

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.convolucion import convolucion_fft
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
