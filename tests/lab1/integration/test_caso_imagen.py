import numpy as np
import pytest

from psi_lab.lab1.caso_imagen import (
    agregar_interferencia,
    aplicar_mascara,
    detectar_picos,
    dft2,
    notch_gaussiano,
    notch_ideal,
    psnr,
)


def imagen_limpia(M=128):
    y, x = np.mgrid[:M, :M]
    img = 0.2 + 0.5 * np.exp(-((x - 45) ** 2 + (y - 60) ** 2) / (2 * 15**2))
    img[80:110, 70:115] = 0.8
    return img


@pytest.mark.parametrize(
    "notch",
    [lambda forma, p: notch_ideal(forma, p, 3), lambda forma, p: notch_gaussiano(forma, p, 2)],
    ids=["ideal", "gaussiano"],
)
def test_flujo_una_interferencia(notch):
    limpia = imagen_limpia()
    sucia = agregar_interferencia(limpia, 20, 12, 0.3)

    picos = detectar_picos(dft2(sucia), 0.5)
    assert {tuple(p) for p in picos} == {(12, 20), (116, 108)}

    filtrada = aplicar_mascara(sucia, notch(sucia.shape, picos))
    assert np.allclose(filtrada.imag, 0)

    antes = psnr(limpia, sucia)
    despues = psnr(limpia, filtrada.real)
    assert despues > 40
    assert despues - antes > 25


def test_flujo_dos_interferencias():
    limpia = imagen_limpia()
    sucia = agregar_interferencia(limpia, 20, 12, 0.3)
    sucia = agregar_interferencia(sucia, 7, 40, 0.2)

    picos = detectar_picos(dft2(sucia), 0.3)
    assert len(picos) == 4

    filtrada = aplicar_mascara(sucia, notch_gaussiano(sucia.shape, picos, 2))
    assert np.allclose(filtrada.imag, 0)
    assert psnr(limpia, filtrada.real) > 40


def test_notch_con_un_solo_pico_deja_parte_imaginaria():
    limpia = imagen_limpia()
    sucia = agregar_interferencia(limpia, 20, 12, 0.3)
    picos = detectar_picos(dft2(sucia), 0.5)

    filtrada = aplicar_mascara(sucia, notch_gaussiano(sucia.shape, picos[:1], 2))
    assert np.abs(filtrada.imag).max() > 0.05
