"""Comando psi-lab1: regenera todas las figuras y tablas del informe."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.muestreo import muestrear


def figura_ejemplo(destino):
    """Figura de relleno para probar la tuberia. Reemplazar por las del informe."""
    t, x = muestrear(lambda t: np.sin(2 * np.pi * 5 * t), fs=100, duracion=1)
    fig = Figure(figsize=(6, 3))
    ax = fig.subplots()
    ax.plot(t, x, ".-")
    ax.set(xlabel="t [s]", ylabel="x(t)", title="Figura de ejemplo")
    fig.tight_layout()
    fig.savefig(destino, dpi=120)

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="psi-lab1", description="Regenera las figuras y tablas del informe."
    )
    parser.add_argument("-o", "--salida", default="out", help="carpeta de salida")
    parser.add_argument(
        "--sintetico",
        action="store_true",
        help="usa solo senales sinteticas, sin descargar datos (CI)",
    )
    args = parser.parse_args(argv)

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    figura_ejemplo(salida / "ejemplo.png")
    # Pendiente: agregar las figuras y tablas de la parte comun
    # Pendiente descargar datos reales (datos.py) y generar las figuras del caso aplicado.

    print("Resultados en", salida)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())