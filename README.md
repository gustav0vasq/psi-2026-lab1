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
El caso real (C4, `fig_caso_real.py`) usa dos fotos con patrón periódico real en `data/caso_real/` (`diario.jpg`, `cielo.jpg`).

## Interfaz pública (módulo, firma, unidades, forma)
| Módulo | Función | Argumentos | Retorna |
|---|---|---|---|
| `psi_lab.lab1.muestreo` | `muestrear(fun, fs, duracion)` | fun: callable de t [s]; fs [Hz]; duracion [s] | `(t, x)` arreglos 1D de largo `round(fs*duracion)` |
| `psi_lab.lab1.muestreo` | `frecuencia_alias(f, fs)` | f [Hz], fs [Hz] (escalares o arreglos) | frecuencia aparente [Hz] en `[0, fs/2]`: `\|f - fs*round(f/fs)\|` |
| `psi_lab.lab1.dft` | `dft(x)` | x: 1D real o complejo, largo N | `X` complejo 1D de largo N, `X[k] = sum_n x[n] exp(-2j pi k n / N)`, sin normalizar (O(N^2)) |
| `psi_lab.lab1.dft` | `idft(X)` | X: 1D complejo, largo N | `x` complejo 1D de largo N, incluye el factor `1/N`; para señales reales usar `.real` |
| `psi_lab.lab1.dft` | `espectro(x, fs, ventana="rect", nfft=None)` | x: 1D real de largo N; fs [Hz]; ventana: `"rect"`, `"hann"` o `"hamming"`; nfft >= N (rellena con ceros; por defecto N) | `(freqs, amp)` 1D de largo `nfft // 2 + 1`: freqs [Hz] = `k fs / nfft` en `[0, fs/2]`; amplitud de un lado (un tono `A cos(2 pi f t)` en un bin da un pico de altura A) |
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
| `psi_lab.lab1.filtros` | `filtrar(x, h)` | x: 1D; h: 1D | `y` 1D del largo de x, sin retardo (compensa `(n_coef - 1) // 2` muestras); usa `convolucion_fft` |
| `psi_lab.lab1.ventanas` (reexportada en `filtros`) | `crear_ventana(nombre, n_coef)` | nombre: `"rect"`, `"hann"`, `"hamming"`, `"blackman"` | ventana simétrica 1D de largo n_coef |
| `psi_lab.lab1.filtros` | `medir_pasabajos(h, fs, fc, n=32768)` | h: pasabajos; fs, fc [Hz] | `(ancho_transicion [Hz], atenuacion [dB])`, medidos entre `1 - delta` y `delta` |
| `psi_lab.lab1.convolucion` | `convolucion_directa(x, h)` | x, h: 1D (listas o arreglos) | `y` 1D de largo `len(x) + len(h) - 1`, convolución lineal por definición (dos ciclos) |
| `psi_lab.lab1.convolucion` | `convolucion_fft(x, h)` | igual que la directa | mismo `y`, con FFT radix-2 propia: rellena con ceros hasta una potencia de 2 >= `len(x) + len(h) - 1`, multiplica los espectros y recorta (sin relleno sale la circular) |
| `psi_lab.lab1.convolucion` | `fft_radix2(x)` | x: 1D de largo potencia de 2 (`ValueError` si no) | `X` complejo 1D, igual que `dft(x)` pero O(N log N) (Cooley-Tukey) |
| `psi_lab.lab1.convolucion` | `ifft_radix2(X)` | X: 1D de largo potencia de 2 | `x` complejo 1D, incluye el factor `1/N` |
| `psi_lab.lab1.convolucion` | `convolucion_separable_2d(img, h_filas, h_cols)` | img: 2D `(alto, ancho)`; h_filas, h_cols: 1D | 2D de `(alto + len(h_cols) - 1, ancho + len(h_filas) - 1)`: h_filas por cada fila y h_cols por cada columna, igual que convolucionar con `np.outer(h_cols, h_filas)` |
| `psi_lab.lab1.caso_imagen` | `dft2(img)` | img: float64 `(alto, ancho)` | `E` complejo `(alto, ancho)`, sin shift, DC en `E[0, 0]` |
| `psi_lab.lab1.caso_imagen` | `idft2(E)` | E: complejo `(alto, ancho)` sin shift | complejo `(alto, ancho)`; la imagen es `.real` |
| `psi_lab.lab1.caso_imagen` | `espectro_log(E)` | E: complejo sin shift | float `(alto, ancho)`, `log(1 + \|E\|)` **ya centrada** (DC en `(alto//2, ancho//2)`); solo para graficar |
| `psi_lab.lab1.caso_imagen` | `fase(E)` | E: complejo sin shift | float `(alto, ancho)`, fase en radianes en `[-pi, pi]`, sin shift |
| `psi_lab.lab1.caso_imagen` | `coseno_2d(forma, ku, kv, amplitud)` | forma: `(alto, ancho)`; ku, kv: ciclos **enteros** sobre la imagen (columnas y filas); amplitud | float `(alto, ancho)`, `amplitud cos(2 pi (ku x / ancho + kv y / alto))`; picos exactos en `(kv, ku)` y `(-kv, -ku)` (mod forma) |
| `psi_lab.lab1.caso_imagen` | `agregar_interferencia(img, ku, kv, amplitud)` | img: 2D; ku, kv, amplitud como en `coseno_2d` | float, `img + coseno_2d(img.shape, ku, kv, amplitud)`, sin recortar a [0, 1] |
| `psi_lab.lab1.caso_imagen` | `detectar_picos(E, umbral, *, radio_dc=10)` | E: complejo `(alto, ancho)` sin shift; umbral: fracción en (0, 1] del máximo de \|E\| fuera del DC; radio_dc [bins]: disco excluido alrededor del DC | int `(n, 2)` `(fila, columna)` sin shift, con **ambos** picos de cada par conjugado; `(0, 2)` si no queda nada fuera del disco |
| `psi_lab.lab1.caso_imagen` | `notch_ideal(forma, picos, radio)` | forma: `(alto, ancho)`; picos: `(n, 2)` int `(fila, columna)` sin shift, **ambos** picos de cada par conjugado; radio [bins] >= 0 | `H` float `(alto, ancho)` en {0, 1}: 0 a distancia **periódica** <= radio de algún pico, 1 fuera |
| `psi_lab.lab1.caso_imagen` | `notch_gaussiano(forma, picos, sigma)` | igual; sigma [bins] > 0 | `H = prod_i (1 - exp(-d_i^2 / 2 sigma^2))` en [0, 1], 0 exacto en cada pico |
| `psi_lab.lab1.caso_imagen` | `aplicar_mascara(img, H)` | img: 2D float; H: misma forma, sin shift | complejo `idft2(dft2(img) * H)`; la imagen es `.real` (imag ~0 si H tiene pares completos) |
| `psi_lab.lab1.caso_imagen` | `psnr(a, b, rango=1.0)` | a, b: misma forma; rango: valor máximo (1.0 en [0, 1]) | PSNR [dB] `= 10 log10(rango^2 / MSE)`; `inf` si son idénticas |
| `psi_lab.lab1.caso_imagen` | `pasabajos_gaussiano(img, sigma)` | img: 2D float; sigma [px] > 0 | imagen 2D de la misma forma: convolución separable con un núcleo gaussiano normalizado (largo `2 ceil(3 sigma) + 1`), bordes por reflejo; ganancia 1 en DC |
| `psi_lab.lab1.caso_imagen` | `zone_plate(M)` | M: lado en px (entero) | float `(M, M)` en [0, 1], `(1 + cos(pi (x^2 + y^2) / M)) / 2` con x, y centradas; frecuencia local `r / M` ciclos/px, llega a 0.5 en el borde |
| `psi_lab.lab1.caso_imagen` | `reducir(img, k)` | img: 2D; k: entero >= 1 | `img[::k, ::k]` (1 de cada k px), **sin** filtrar antes |
| `psi_lab.lab1.caso_imagen` | `pasabajos_separable_2d(img, fc, n_coef)` | img: 2D float; fc [ciclos/px] en (0, 0.5); n_coef: entero | imagen 2D de la misma forma: `fir_pasabajos(fc, 1, n_coef)` por filas y columnas, recortada al centro; ganancia 1 en DC, bordes oscurecidos ~n_coef/2 px por el relleno con ceros |
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

## Estructura
`src/psi_lab/lab1/` (código; `caso_imagen/` con un archivo por tema del caso C; `figuras/` con un archivo `fig_*.py` por figura; `ventanas.py` con las ventanas compartidas), `tests/lab1/{unit,integration}/`, `informe/lab1.pdf`.