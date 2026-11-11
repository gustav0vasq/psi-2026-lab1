"""Figuras y tabla de la Tarea 4 (filtros FIR por ventana). Dueno: R3."""

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.figuras import figura, guardar_tabla
from psi_lab.lab1.filtros import (
    VENTANAS,
    filtrar,
    fir_pasaaltos,
    fir_pasabajos,
    fir_pasabanda,
    fir_rechazabanda,
    medir_pasabajos,
    respuesta_frecuencia,
)
from psi_lab.lab1.sinteticos import chirp

FS = 1000.0  # [Hz]
FC = 100.0  # corte del pasabajos de referencia [Hz]
N_COEF = 101
N_COEFS_TABLA = (21, 51, 101, 201)  # los del enunciado (Tarea 4.2)
VENTANAS_FIG = ("rect", "hamming", "hann")  # las del enunciado; la tabla agrega blackman


def _db(H):
    return 20 * np.log10(np.maximum(np.abs(H), 1e-8))


@figura("filtros_tipos")
def _(salida):
    filtros = {
        f"pasabajos fc={FC:g} Hz": fir_pasabajos(FC, FS, N_COEF),
        f"pasaaltos fc={FC:g} Hz": fir_pasaaltos(FC, FS, N_COEF),
        "pasabanda 150-250 Hz": fir_pasabanda(150, 250, FS, N_COEF),
        "rechazabanda 150-250 Hz": fir_rechazabanda(150, 250, FS, N_COEF),
    }
    fig = Figure(figsize=(9, 6))
    axs = fig.subplots(2, 2, sharex=True, sharey=True).ravel()
    for ax, (nombre, h) in zip(axs, filtros.items()):
        freqs, H = respuesta_frecuencia(h, FS)
        ax.plot(freqs, _db(H), lw=0.9)
        ax.set(title=nombre, ylim=(-100, 5))
        ax.grid(alpha=0.3)
    for ax in axs[2:]:
        ax.set_xlabel("f [Hz]")
    for ax in axs[::2]:
        ax.set_ylabel("|H(f)| [dB]")
    fig.suptitle(f"Filtros derivados del pasabajos (Hamming, n_coef={N_COEF}, fs={FS:g} Hz)")
    fig.tight_layout()
    fig.savefig(salida / "filtros_tipos.png", dpi=120)


@figura("filtros_ventanas")
def _(salida):
    fig = Figure(figsize=(9, 4))
    ax_h, ax_f = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
    for v in VENTANAS:
        h = fir_pasabajos(FC, FS, N_COEF, v)
        freqs, H = respuesta_frecuencia(h, FS)
        ax_h.plot(np.arange(N_COEF) - N_COEF // 2, h, lw=0.9, label=v)
        ax_f.plot(freqs, _db(H), lw=0.9, label=v)
    ax_h.set(xlabel="n - M", ylabel="h[n]", title="Respuesta al impulso")
    ax_f.set(xlabel="f [Hz]", ylabel="|H(f)| [dB]", ylim=(-120, 5),
             title=f"Pasabajos fc={FC:g} Hz, n_coef={N_COEF}")
    ax_f.legend()
    for ax in (ax_h, ax_f):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(salida / "filtros_ventanas.png", dpi=120)


@figura("filtros_n_coef")
def _(salida):
    fig = Figure(figsize=(12, 4))
    axs = fig.subplots(1, len(VENTANAS_FIG), sharey=True)
    for ax, v in zip(axs, VENTANAS_FIG):
        for n in N_COEFS_TABLA:
            freqs, H = respuesta_frecuencia(fir_pasabajos(FC, FS, n, v), FS)
            ax.plot(freqs, _db(H), lw=0.9, label=f"n_coef={n}")
        ax.axvline(FC, color="k", ls=":", lw=0.8)
        ax.set(xlabel="f [Hz]", title=f"ventana {v}", ylim=(-110, 5))
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("|H(f)| [dB]")
    axs[-1].legend(fontsize=8)
    fig.suptitle(f"Pasabajos fc={FC:g} Hz, fs={FS:g} Hz: n_coef fija el ancho de transicion, "
                 "la ventana la atenuacion", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida / "filtros_n_coef.png", dpi=120)

    filas = []
    for v in VENTANAS:
        for n in N_COEFS_TABLA:
            ancho, aten = medir_pasabajos(fir_pasabajos(FC, FS, n, v), FS, FC)
            filas.append([v, n, f"{ancho:.2f}", f"{ancho * n / FS:.2f}", f"{aten:.1f}"])
    guardar_tabla(
        salida,
        "filtros_transicion_atenuacion",
        ["ventana", "n_coef", "ancho_transicion_hz", "ancho_por_n_sobre_fs", "atenuacion_db"],
        filas,
    )


@figura("filtros_chirp")
def _(salida):
    """Tarea 4.3: un chirp de 0 a 500 Hz filtrado por el pasabajos de 100 Hz.

    La frecuencia instantanea crece linealmente, f(t) = 500 t / duracion, asi que el eje del
    tiempo es tambien un eje de frecuencia: la salida conserva la amplitud mientras f < fc y se
    apaga en la transicion, que en el tiempo dura (ancho de transicion) * duracion / 500.
    """
    duracion = 1.0
    t = np.arange(round(FS * duracion)) / FS
    x = chirp(0, FS / 2, FS, duracion)
    h = fir_pasabajos(FC, FS, N_COEF)
    y = filtrar(x, h)
    ancho, _ = medir_pasabajos(h, FS, FC)
    t_corte = FC * duracion / (FS / 2)

    fig = Figure(figsize=(10, 5.5))
    ax_x, ax_y = fig.subplots(2, 1, sharex=True)
    ax_x.plot(t, x, lw=0.4)
    ax_x.set(ylabel="x(t)", title=f"Entrada: chirp de 0 a {FS / 2:g} Hz en {duracion:g} s")
    ax_y.plot(t, y, lw=0.4, label="salida")
    ax_y.axvspan(t_corte - ancho / 2 * duracion / (FS / 2),
                 t_corte + ancho / 2 * duracion / (FS / 2),
                 color="tab:orange", alpha=0.25, label=f"transicion (~{ancho:.0f} Hz)")
    ax_y.set(xlabel="t [s]", ylabel="y(t)",
             title=f"Salida del pasabajos fc={FC:g} Hz (Hamming, n_coef={N_COEF})")
    ax_y.legend(loc="upper right", fontsize=8)
    for ax in (ax_x, ax_y):
        ax.axvline(t_corte, color="k", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
    sec = ax_x.secondary_xaxis("top", functions=(lambda tt: tt * FS / 2 / duracion,
                                                lambda f: f * duracion / (FS / 2)))
    sec.set_xlabel("frecuencia instantanea [Hz]")
    fig.tight_layout()
    fig.savefig(salida / "filtros_chirp.png", dpi=120)
