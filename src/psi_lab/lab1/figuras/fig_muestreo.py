"""Tarea 1: tabla de frecuencias aparentes y dos frecuencias con las mismas muestras."""

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.dft import espectro
from psi_lab.lab1.figuras import figura, guardar_tabla
from psi_lab.lab1.muestreo import frecuencia_alias, muestrear
from psi_lab.lab1.sinteticos import tono

FS = 10
FRECUENCIAS = (3, 7, 13, 23)
DURACION = 10  # s: resolucion 1/DURACION = 0.1 Hz, y los tonos de la tabla caen en un bin
FS_FINO = 1000  # "continua": muestreada mucho mas fino que cualquier f de la figura


@figura("muestreo_alias")
def _(salida):
    filas = []
    for f in FRECUENCIAS:
        freqs, amp = espectro(tono(f, FS, DURACION), FS)
        medida = float(freqs[np.argmax(amp)])
        predicha = float(frecuencia_alias(f, FS))
        filas.append([f, FS, predicha, medida, abs(medida - predicha)])
    guardar_tabla(
        salida,
        "muestreo_alias",
        ["f_hz", "fs_hz", "f_alias_predicha_hz", "f_medida_hz", "error_hz"],
        filas,
    )

    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    f1, f2 = 3, 13  # f2 = f1 + fs: mismas muestras
    for f, estilo in ((f1, "-"), (f2, "--")):
        t_fino, x_fino = muestrear(lambda t, f=f: np.cos(2 * np.pi * f * t), FS_FINO, 1)
        ax.plot(t_fino, x_fino, estilo, label=f"{f} Hz")
    t, x = muestrear(lambda t: np.cos(2 * np.pi * f1 * t), FS, 1)
    ax.plot(t, x, "ko", label=f"muestras (fs = {FS} Hz)")
    ax.set(xlabel="t [s]", ylabel="x(t)", title="Dos frecuencias, las mismas muestras")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(salida / "muestreo_alias.png", dpi=120)
