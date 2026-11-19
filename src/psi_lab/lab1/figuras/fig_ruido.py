"""Tarea C2: ruido periodico sintetico y deteccion automatica de sus picos. Dueno: R2."""

from matplotlib.figure import Figure
from skimage import data

from psi_lab.lab1.caso_imagen import agregar_interferencia, detectar_picos, dft2, espectro_log
from psi_lab.lab1.figuras import figura, guardar_tabla

# Interferencias (amplitud, ku, kv): ciclos enteros sobre la imagen de 512 x 512
INTERFERENCIAS = ((0.2, 40, 25), (0.12, 70, -50))
UMBRAL = 0.3


def _imagen():
    """camera de skimage.data (viene con el paquete, no se descarga), float en [0, 1]."""
    return data.camera() / 255.0


def _con_ruido(img, interferencias):
    for amplitud, ku, kv in interferencias:
        img = agregar_interferencia(img, ku, kv, amplitud)
    return img


def _picos_esperados(forma, interferencias):
    """Bins (v, u) donde la teoria pone los picos: (kv, ku) y su conjugado, sin shift."""
    alto, ancho = forma
    esperados = set()
    for _, ku, kv in interferencias:
        esperados |= {(kv % alto, ku % ancho), (-kv % alto, -ku % ancho)}
    return esperados


@figura("ruido_deteccion")
def _(salida):
    img = _imagen()
    ruidosa = _con_ruido(img, INTERFERENCIAS)
    E = dft2(ruidosa)
    picos = detectar_picos(E, UMBRAL)
    alto, ancho = img.shape

    fig = Figure(figsize=(11, 4), layout="constrained")
    ax_l, ax_r, ax_e = fig.subplots(1, 3)
    ax_l.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax_l.set_title("Original", fontsize=9)
    ax_r.imshow(ruidosa, cmap="gray", vmin=0, vmax=1)
    ax_r.set_title("Con 2 interferencias A cos 2π(u₀x + v₀y)", fontsize=9)
    ax_e.imshow(espectro_log(E), cmap="gray")
    # Los picos vienen sin shift; el espectro se dibuja centrado: se corren medio tamano
    filas = (picos[:, 0] + alto // 2) % alto
    columnas = (picos[:, 1] + ancho // 2) % ancho
    ax_e.scatter(columnas, filas, s=120, facecolors="none", edgecolors="red", linewidths=1.2)
    ax_e.set_title(f"log(1 + |E|) centrado: {len(picos)} picos detectados", fontsize=9)
    for ax in (ax_l, ax_r, ax_e):
        ax.axis("off")
    fig.savefig(salida / "ruido_deteccion.png", dpi=120)


@figura("ruido_sensibilidad")
def _(salida):
    """Tabla: cuantos picos acierta el detector segun la amplitud y el radio excluido."""
    img = _imagen()
    filas = []
    for amplitud in (0.2, 0.1, 0.05, 0.02, 0.01):
        interferencias = ((amplitud, 40, 25), (0.6 * amplitud, 70, -50))
        esperados = _picos_esperados(img.shape, interferencias)
        E = dft2(_con_ruido(img, interferencias))
        for radio_dc in (4, 10):
            hallados = {tuple(int(c) for c in p) for p in detectar_picos(E, UMBRAL, radio_dc=radio_dc)}
            filas.append([amplitud, radio_dc, len(hallados & esperados), len(esperados),
                          len(hallados - esperados)])
    guardar_tabla(salida, "ruido_sensibilidad",
                  ["amplitud", "radio_dc", "aciertos", "esperados", "falsos"], filas)