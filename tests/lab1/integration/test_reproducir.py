from psi_lab.lab1.reproducir import main


def test_reproducir_sintetico_genera_figuras(tmp_path):
    assert main(["--sintetico", "-o", str(tmp_path)]) == 0
    figuras = list(tmp_path.glob("*.png"))
    assert figuras and all(f.stat().st_size > 0 for f in figuras)