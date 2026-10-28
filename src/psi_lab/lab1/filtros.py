"""Parte comun 4: filtros FIR por ventana."""


# Por implementar:
def fir_pasabajos(fc, fs, n_coef, ventana="hamming"):
    """Sinc truncada y enventanada, normalizada a ganancia 1 en DC."""
    raise NotImplementedError

def fir_pasaaltos(fc, fs, n_coef, ventana="hamming"):
    raise NotImplementedError

def fir_pasabanda(f1, f2, fs, n_coef, ventana="hamming"):
    raise NotImplementedError

def fir_rechazabanda(f1, f2, fs, n_coef, ventana="hamming"):
    raise NotImplementedError

def respuesta_frecuencia(h, fs, n=4096):
    """Retorna (freqs, H) con H complejo."""
    raise NotImplementedError

def filtrar(x, h):
    """Aplica h a x (largo igual a x), compensando el retardo (n_coef - 1) / 2."""
    raise NotImplementedError