"""Los contratos del caso C no pueden cambiar sin avisar al grupo."""
import importlib
import inspect

import pytest

INTERFAZ = {
    "fourier2d": {"dft2": ["img"], "idft2": ["E"], "espectro_log": ["E"], "fase": ["E"]},
    "ruido": {
        "coseno_2d": ["forma", "ku", "kv", "amplitud"],
        "agregar_interferencia": ["img", "ku", "kv", "amplitud"],
        "detectar_picos": ["E", "umbral"],
    },
    "notch": {
        "notch_ideal": ["forma", "picos", "radio"],
        "notch_gaussiano": ["forma", "picos", "sigma"],
        "aplicar_mascara": ["img", "H"],
        "pasabajos_gaussiano": ["img", "sigma"],
    },
    "metricas": {"psnr": ["a", "b", "rango"]},
    "aliasing": {
        "zone_plate": ["M"],
        "reducir": ["img", "k"],
        "pasabajos_separable_2d": ["img", "fc", "n_coef"],
    },
}
CASOS = [(m, f) for m, funcs in INTERFAZ.items() for f in funcs]
PAQUETE = "psi_lab.lab1.caso_imagen"


@pytest.mark.parametrize("modulo,funcion", CASOS)
def test_firma(modulo, funcion):
    mod = importlib.import_module(f"{PAQUETE}.{modulo}")
    esperado = INTERFAZ[modulo][funcion]
    params = list(inspect.signature(getattr(mod, funcion)).parameters)
    assert params[: len(esperado)] == esperado


@pytest.mark.parametrize("modulo,funcion", CASOS)
def test_paquete_reexporta(modulo, funcion):
    paquete = importlib.import_module(PAQUETE)
    mod = importlib.import_module(f"{PAQUETE}.{modulo}")
    assert getattr(paquete, funcion) is getattr(mod, funcion)
