"""Las firmas obligatorias del enunciado no pueden cambiar."""
import importlib
import inspect

import pytest

INTERFAZ = {
    "muestreo": {"muestrear": ["fun", "fs", "duracion"], "frecuencia_alias": ["f", "fs"]},
    "dft": {
        "dft": ["x"],
        "idft": ["X"],
        "espectro": ["x", "fs", "ventana", "nfft"],
    },
    "convolucion": {"convolucion_directa": ["x", "h"], "convolucion_fft": ["x", "h"]},
    "filtros": {
        "fir_pasabajos": ["fc", "fs", "n_coef", "ventana"],
        "fir_pasaaltos": ["fc", "fs", "n_coef", "ventana"],
        "fir_pasabanda": ["f1", "f2", "fs", "n_coef", "ventana"],
        "fir_rechazabanda": ["f1", "f2", "fs", "n_coef", "ventana"],
        "respuesta_frecuencia": ["h", "fs", "n"],
        "filtrar": ["x", "h"],
    },
}
CASOS = [(m, f) for m, funcs in INTERFAZ.items() for f in funcs]

@pytest.mark.parametrize("modulo,funcion", CASOS)
def test_firma(modulo, funcion):
    mod = importlib.import_module(f"psi_lab.lab1.{modulo}")
    esperado = INTERFAZ[modulo][funcion]
    params = list(inspect.signature(getattr(mod, funcion)).parameters)
    assert params[: len(esperado)] == esperado