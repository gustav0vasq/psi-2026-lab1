"""Figuras y tablas de la Tarea 2: DFT, resolucion y fuga espectral. Dueno: R2."""

import time

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.dft import dft, espectro
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


FS = 1000.0  # [Hz]
NFFT_FINO = 8192  # relleno para dibujar la forma "continua" del espectro


def _picos(f, a, f_min, f_max, fraccion=0.5):
    """Frecuencias de los maximos locales en [f_min, f_max] que superan fraccion * maximo."""
    m = (f >= f_min) & (f <= f_max)
    f, a = f[m], a[m]
    es_pico = (a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > fraccion * a.max())
    return f[1:-1][es_pico]


@figura("dft_resolucion")
def _(salida):
    """Tarea 2.3: dos tonos de 50 y 52 Hz con duraciones T; se separan si 1/T <= 2 Hz."""
    duraciones = (0.25, 0.5, 1.0, 2.0)
    fig = Figure(figsize=(10, 6))
    axs = fig.subplots(2, 2, sharey=True).ravel()
    filas = []
    for ax, T in zip(axs, duraciones):
        N = round(FS * T)
        t = np.arange(N) / FS
        x = np.cos(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 52 * t)
        f_bins, a_bins = espectro(x, FS)  # los N bins reales, cada 1/T Hz
        f_fino, a_fino = espectro(x, FS, nfft=NFFT_FINO) 
        ax.plot(f_fino, a_fino, lw=0.9, label="forma (relleno a 8192)")
        ax.plot(f_bins, a_bins, "o", ms=5, label=f"bins de la DFT (cada {1 / T:g} Hz)")
        for f0 in (50, 52):
            ax.axvline(f0, color="k", ls=":", lw=0.8)
        picos = _picos(f_fino, a_fino, 47, 55)
        separados = len(picos) >= 2
        ax.set(xlim=(40, 62), title=f"T = {T:g} s, N = {N}, Δf = 1/T = {1 / T:g} Hz"
               f" → {'se separan' if separados else 'NO se separan'}")
        ax.grid(alpha=0.3)
        filas.append([T, N, f"{1 / T:g}", "si" if separados else "no"])
    for ax in axs[2:]:
        ax.set_xlabel("f [Hz]")
    for ax in axs[::2]:
        ax.set_ylabel("amplitud")
    axs[0].legend(fontsize=8)
    fig.suptitle("Resolucion: tonos de 50 y 52 Hz (fs = 1000 Hz, ventana rectangular)")
    fig.tight_layout()
    fig.savefig(salida / "dft_resolucion.png", dpi=120)
    guardar_tabla(salida, "dft_resolucion", ["T_s", "N", "delta_f_hz", "se_separan"], filas)


def _dos_tonos(T):
    """Tonos de 50 y 52 Hz de amplitud 1 durante T segundos a FS."""
    t = np.arange(round(FS * T)) / FS
    return np.cos(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 52 * t)


@figura("dft_relleno")
def _(salida):
    """T = 0,25 s con nfft = 8N. El relleno interpola, no mejora la resolucion."""
    x = _dos_tonos(0.25)
    N = x.size
    f1, a1 = espectro(x, FS)  # bins cada 4 Hz
    f8, a8 = espectro(x, FS, nfft=8 * N)  # bins cada 0,5 Hz
    f_ref, a_ref = espectro(_dos_tonos(1.0), FS)  # mas duracion: bins cada 1 Hz

    fig = Figure(figsize=(10, 4))
    ax_r, ax_t = fig.subplots(1, 2, sharey=True)
    ax_r.plot(f8, a8, ".-", ms=4, lw=0.8, label=f"nfft = 8N = {8 * N} (cada {FS / (8 * N):g} Hz)")
    ax_r.plot(f1, a1, "o", ms=7, label=f"nfft = N = {N} (cada {FS / N:g} Hz)")
    ax_r.set(title="T = 0,25 s: con relleno sigue habiendo un solo lobulo")
    ax_t.plot(f_ref, a_ref, ".-", ms=4, lw=0.8, color="tab:green",
              label="T = 1 s, nfft = N = 1000 (cada 1 Hz)")
    ax_t.set(title="T = 1 s sin relleno: los tonos se separan")
    for ax in (ax_r, ax_t):
        for f0 in (50, 52):
            ax.axvline(f0, color="k", ls=":", lw=0.8)
        ax.set(xlim=(40, 62), xlabel="f [Hz]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    ax_r.set_ylabel("amplitud")
    fig.suptitle("Relleno con ceros: mas puntos, misma resolucion (solo depende de T)")
    fig.tight_layout()
    fig.savefig(salida / "dft_relleno.png", dpi=120)


@figura("dft_fuga")
def _(salida):
    """Tono de 50 Hz (A = 1) y de 80 Hz (A = 1e-3). Rectangular vs. Hann en dB.

    N = 1024 hace que 50 y 80 Hz NO caigan en un bin (bins cada 0,977 Hz), que es lo normal
    en una senal real: asi aparece la fuga. Con N = 1000 ambos tonos caerian justo en un bin
    y la rectangular no mostraria fuga, lo que esconderia el efecto que se quiere estudiar.
    """
    N = 1024
    t = np.arange(N) / FS
    x = np.cos(2 * np.pi * 50 * t) + 1e-3 * np.cos(2 * np.pi * 80 * t)

    fig = Figure(figsize=(9, 4.5))
    ax = fig.subplots()
    filas = []
    for ventana in ("rect", "hann"):
        f, a = espectro(x, FS, ventana=ventana)
        db = 20 * np.log10(np.maximum(a, 1e-12))
        ax.plot(f, db, lw=0.9, label=ventana)
        cerca_80 = np.abs(f - 80) <= 1
        fondo = (np.abs(f - 80) > 3) & (np.abs(f - 80) < 10)  # alrededor del tono debil
        filas.append([ventana, f"{db[cerca_80].max():.1f}", f"{db[fondo].max():.1f}"])
    ax.axhline(-60, color="k", ls=":", lw=0.8, label="nivel del tono debil (-60 dB)")
    ax.annotate("tono de 80 Hz\n(A = 10⁻³)", xy=(80, -60), xytext=(110, -30),
                arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set(xlim=(0, 200), ylim=(-120, 5), xlabel="f [Hz]", ylabel="amplitud [dB]",
           title=f"Fuga espectral: 50 Hz (A = 1) + 80 Hz (A = 10⁻³), N = {N}, fs = {FS:g} Hz")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(salida / "dft_fuga.png", dpi=120)
    guardar_tabla(salida, "dft_fuga", ["ventana", "pico_cerca_80hz_db", "fondo_alrededor_db"], filas)