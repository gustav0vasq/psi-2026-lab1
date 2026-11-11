# PSI 2-2026 - Laboratorio 1: del muestreo al filtrado

![CI](https://github.com/gustav0vasq/psi-2026-lab1/actions/workflows/ci.yml/badge.svg)

## Instalación y uso sin Docker
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,caso]"
pytest -v
psi-lab1            # regenera figuras y tablas en out/
```

## Uso con Docker
```bash
docker compose build
docker compose run --rm tests   # pruebas unitarias y de integracion
docker compose run --rm lab1    # reproduce todas las figuras y tablas en out/
```

## Datos
No se versionan. `datos.py` los descarga o genera (carpeta `data/`, o `PSI_DATOS`).
El CI usa solo señales sintéticas (`psi-lab1 --sintetico`).

## Interfaz pública (módulo, firma, unidades, forma)
| Módulo | Función | Argumentos | Retorna |
|---|---|---|---|
| `psi_lab.lab1.muestreo` | `muestrear(fun, fs, duracion)` | fun: callable de t [s]; fs [Hz]; duracion [s] | `(t, x)` arreglos 1D de largo `round(fs*duracion)` |
| `psi_lab.lab1.sinteticos` | `tono(f, fs, duracion)` | f [Hz]; fs [Hz]; duracion [s] | `x` 1D de largo `round(fs*duracion)`, `cos(2 pi f t)` de amplitud 1 |
| `psi_lab.lab1.sinteticos` | `suma_tonos(frecuencias, fs, duracion)` | frecuencias: secuencia de f [Hz]; fs [Hz]; duracion [s] | `x` 1D, suma de tonos de amplitud 1 |
| `psi_lab.lab1.sinteticos` | `chirp(f0, f1, fs, duracion)` | f0, f1 [Hz]; fs [Hz]; duracion [s] | `x` 1D, barrido lineal de f0 a f1, amplitud 1 |
| `psi_lab.lab1.sinteticos` | `ruido_a_snr(x, snr_db, semilla=0)` | x: arreglo 1D; snr_db [dB]; semilla: entero | `x + ruido` blanco gaussiano con SNR exacto `10 log10(P_x / P_ruido) = snr_db`; reproducible por semilla |
| `psi_lab.lab1.figuras` | `figura(nombre, requiere_datos=False)` | nombre: str único; requiere_datos: bool | decorador que registra `fun(salida)`; `ValueError` si el nombre ya existe |
| `psi_lab.lab1.figuras` | `guardar_tabla(salida, nombre, encabezado, filas)` | salida: carpeta (Path); nombre: str; encabezado: lista de str; filas: lista de listas | ruta `salida/tablas/<nombre>.csv` (Path) |
| `psi_lab.lab1.figuras` | `generar(salida, sintetico=False)` | salida: carpeta (Path); sintetico: bool | lista de nombres de las figuras generadas; con `sintetico=True` omite las que requieren datos |
| `psi_lab.lab1.filtros` | `fir_pasabajos(fc, fs, n_coef, ventana="hamming")` | fc [Hz] en (0, fs/2); fs [Hz]; n_coef: entero; ventana: `"rect"`, `"hann"`, `"hamming"` o `"blackman"` | `h` 1D de largo `n_coef`, simétrico (fase lineal), `sum(h) = 1` |
| `psi_lab.lab1.filtros` | `fir_pasaaltos(fc, fs, n_coef, ventana="hamming")` | igual que el pasabajos; `n_coef` **impar** | `h = delta - h_pasabajos`; ganancia 0 en DC |
| `psi_lab.lab1.filtros` | `fir_pasabanda(f1, f2, fs, n_coef, ventana="hamming")` | f1 < f2 [Hz] en (0, fs/2) | `h = h_lp(f2) - h_lp(f1)`; ganancia 0 en DC |
| `psi_lab.lab1.filtros` | `fir_rechazabanda(f1, f2, fs, n_coef, ventana="hamming")` | f1 < f2 [Hz]; `n_coef` **impar** | `h = delta - h_pasabanda`; ganancia 1 en DC |
| `psi_lab.lab1.filtros` | `respuesta_frecuencia(h, fs, n=4096)` | h: 1D; fs [Hz]; n: número de frecuencias | `(freqs, H)`: freqs [Hz] = `i fs / (2n)`, i = 0..n-1 (en [0, fs/2)); `H` complejo de largo n |
| `psi_lab.lab1.filtros` | `filtrar(x, h)` | x: 1D; h: 1D | `y` 1D del largo de x, sin retardo (compensa `(n_coef - 1) // 2` muestras) |
| `psi_lab.lab1.filtros` | `crear_ventana(nombre, n_coef)` | nombre: `"rect"`, `"hann"`, `"hamming"`, `"blackman"` | ventana simétrica 1D de largo n_coef |
| `psi_lab.lab1.filtros` | `medir_pasabajos(h, fs, fc, n=32768)` | h: pasabajos; fs, fc [Hz] | `(ancho_transicion [Hz], atenuacion [dB])`, medidos entre `1 - delta` y `delta` |
| ... | ... | ... | ... |

## Figuras y tablas del informe
Cada figura vive en su propio archivo `src/psi_lab/lab1/figuras/fig_<tema>.py`, de un solo dueño.
`reproducir.py` no se modifica: descubre solo los archivos `fig_*.py` y ejecuta lo registrado,
en orden alfabético por nombre de figura.

```python
from psi_lab.lab1.figuras import figura, guardar_tabla

@figura("notch_comparacion")                 # requiere_datos=True si usa fotos reales
def _(salida):                               # salida: Path de la carpeta de salida
    ...
    fig.savefig(salida / "notch_comparacion.png", dpi=120)
    guardar_tabla(salida, "notch_psnr", ["radio", "psnr"], filas)   # -> out/tablas/notch_psnr.csv
```
- Los nombres de figura son únicos (un duplicado falla al importar).
- Con `psi-lab1 --sintetico` se omiten las figuras con `requiere_datos=True` (así corre el CI).
- Si una figura lanza una excepción, el comando falla: no se ocultan errores.
- `fig_ejemplo.py` es de relleno y se elimina cuando haya figuras reales.

## Estructura
`src/psi_lab/lab1/` (código; `figuras/` con un archivo `fig_*.py` por figura), `tests/lab1/{unit,integration}/`, `informe/lab1.pdf`.