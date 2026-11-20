import numpy as np

from psi_lab.lab1.caso_imagen import pasabajos_separable_2d, reducir, zone_plate


def test_zone_plate():
    z = zone_plate(64)
    assert z.shape == (64, 64)
    assert z.min() >= 0 and z.max() <= 1
    assert np.allclose(z, z.T)
    assert z[32, 32] == 1


def test_reducir():
    rng = np.random.default_rng(0)
    img = rng.random((10, 15))
    r = reducir(img, 3)
    assert r.shape == (4, 5)
    assert r[2, 3] == img[6, 9]


def test_reducir_aliasing():
    n = np.arange(64)
    img = np.tile(np.cos(2 * np.pi * 0.4 * n), (8, 1))
    r = reducir(img, 2)
    m = np.arange(r.shape[1])
    assert np.allclose(r[0], np.cos(2 * np.pi * 0.2 * m))


def test_pasabajos_dc():
    img = np.ones((40, 50))
    f = pasabajos_separable_2d(img, 0.25, 21)
    assert f.shape == img.shape
    assert np.allclose(f[10:30, 10:40], 1)


def test_pasabajos_corte():
    n = np.arange(64)
    baja = np.tile(np.cos(2 * np.pi * 0.05 * n), (64, 1))
    alta = np.tile(np.cos(2 * np.pi * 0.4 * n), (64, 1))
    f_baja = pasabajos_separable_2d(baja, 0.25, 31)
    f_alta = pasabajos_separable_2d(alta, 0.25, 31)
    centro = (slice(16, 48), slice(16, 48))
    assert np.allclose(f_baja[centro], baja[centro], atol=0.02)
    assert np.abs(f_alta[centro]).max() < 0.01
