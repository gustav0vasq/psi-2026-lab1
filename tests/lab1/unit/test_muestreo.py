import numpy as np
import pytest

from psi_lab.lab1.muestreo import muestrear


def test_muestrear_largo_y_paso():
    t, x = muestrear(lambda t: np.sin(2 * np.pi * 5 * t), fs=100, duracion=2)
    assert t.size == x.size == 200
    assert t[1] - t[0] == pytest.approx(1 / 100)