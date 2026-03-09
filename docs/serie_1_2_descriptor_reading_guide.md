# Guia de lectura para `serie_1 (2).mat`

## 1. Que es este dataset

`serie_1 (2).mat` no es una traza unica. Es una matriz de `5000` capturas por `1565` muestras.

- Cada fila se interpreta como una captura independiente.
- El eje horizontal del archivo `time` no es suficientemente confiable para tratarlo como tiempo continuo de muestreo.
- Por eso las graficas de forma de onda se leen contra `sample_index`, y la evolucion fisica se lee principalmente contra `row_idx`.

En este estudio, `row_idx` es el mejor sustituto de "orden del experimento".

## 2. Donde mirar

Los productos principales estan en:

- [series_matrix_overview.png](/e:/DeltaDP/outputs/mat_series_1_2/series_matrix_overview.png)
- [descriptor_trends.png](/e:/DeltaDP/outputs/mat_series_1_2/descriptor_trends.png)
- [descriptor_heatmap.png](/e:/DeltaDP/outputs/mat_series_1_2/descriptor_heatmap.png)
- [representative_waveforms.png](/e:/DeltaDP/outputs/mat_series_1_2/representative_waveforms.png)
- [study_report.md](/e:/DeltaDP/outputs/mat_series_1_2/study_report.md)
- [descriptor_behavior.csv](/e:/DeltaDP/outputs/mat_series_1_2/descriptor_behavior.csv)
- [change_candidates.csv](/e:/DeltaDP/outputs/mat_series_1_2/change_candidates.csv)
- [activity_blocks.csv](/e:/DeltaDP/outputs/mat_series_1_2/activity_blocks.csv)

## 3. Como leer cada grafica

### `series_matrix_overview.png`

Es la vista global de toda la matriz.

- Eje `y`: numero de captura.
- Eje `x`: indice de muestra dentro de cada captura.
- Color: amplitud instantanea.

Que buscar:

- bloques verticales de actividad intensa;
- zonas compactas y consecutivas en filas;
- cambios abruptos en la textura de la matriz.

En este dataset, el bloque dominante esta entre las filas `497` y `535`.

### `descriptor_trends.png`

Resume la evolucion fila a fila de los descriptores.

Paneles:

- `Activity and Change`: muestra el nivel global de actividad y el puntaje de cambio entre capturas consecutivas.
- `Amplitude Descriptors`: muestra energia y amplitudes robustas.
- `Occupancy and Count`: muestra cuanto de la captura esta "ocupado" por actividad y cuantas detecciones aparecen.
- `Temporal and Shape`: muestra estructura de intervalos y forma estadistica.

Que buscar:

- picos anchos: cambio de regimen sostenido;
- picos angostos: transiciones puntuales;
- descriptores que suben juntos: evidencia fuerte de cambio fisico;
- descriptores que suben solos: posible artefacto o cambio parcial.

### `descriptor_heatmap.png`

Es la forma mas rapida de ver combinaciones.

- Cada fila del heatmap es un descriptor estandarizado.
- Rojo: por encima del comportamiento tipico.
- Azul: por debajo del comportamiento tipico.

Que buscar:

- columnas rojas simultaneas en varios descriptores: actividad fisica consistente;
- bandas rojas largas: fase sostenida;
- descriptores que siempre copian a otro: redundancia.

### `representative_waveforms.png`

Compara una captura basal, una de maxima actividad y una posterior.

Que buscar:

- aumento del pico maximo;
- ocupacion mas ancha de la captura;
- aparicion de multiples impulsos en la misma fila;
- cambio de simetria o de cola.

## 4. Que hace cada descriptor

### Amplitud y energia

- `energy_v2`: energia media de la captura. Es el descriptor mas bruto de severidad.
- `rms_v`: amplitud efectiva. Muy parecido a energia.
- `p95_abs_v`: amplitud robusta alta. Es menos sensible que el pico maximo a un solo impulso aislado.
- `peak_abs_v`: el maximo absoluto. Sirve para detectar eventos extremos.
- `mean_pulse_amp_v`: amplitud media de los pulsos detectados. Es util cuando hay muchos impulsos verdaderos y no solo un pico espurio.

Interpretacion:

- si suben `energy_v2`, `rms_v` y `p95_abs_v` juntos, la descarga se intensifico de manera global;
- si solo sube `peak_abs_v`, puede ser un evento aislado;
- si sube `mean_pulse_amp_v`, hay pulsos mas energicos, no solo mas ruido.

### Ocupacion y conteo

- `active_ratio`: fraccion de muestras por encima de un umbral robusto.
- `pulse_count`: numero de pulsos detectados en la captura.
- `pulse_rate_hz`: conteo normalizado por duracion de captura.

Interpretacion:

- `active_ratio` alto significa que la actividad ocupa mas tiempo dentro de la ventana;
- `pulse_count` alto significa mas estructura impulsiva;
- si suben ambos, hay un cambio fuerte y sostenido;
- si sube `pulse_count` pero no `active_ratio`, pueden ser muchos pulsos pequenos o separados.

### Temporales

- `median_dt_s`: separacion tipica entre pulsos.
- `iqr_dt_s`: dispersion robusta de los intervalos.
- `cv_dt`: irregularidad relativa de los intervalos.
- `burstiness`: si la actividad se agrupa en rafagas o es mas uniforme.

Interpretacion:

- `median_dt_s` bajo suele significar pulsos mas compactos;
- `cv_dt` alto significa patron irregular;
- `burstiness` menos negativo o mas positivo indica comportamiento mas agrupado.

En este dataset, los temporales ayudan menos que amplitud y forma para aislar el bloque principal.

### Forma estadistica

- `kurtosis`: mide colas pesadas y presencia de eventos extremos.
- `skewness`: mide asimetria.
- `crest_factor`: relacion entre pico maximo y RMS.

Interpretacion:

- `kurtosis` alta detecta impulsos raros y muy fuertes;
- `crest_factor` alto detecta capturas muy picudas;
- si `kurtosis` sube sin que suba mucho la energia, puede haber outliers impulsivos;
- si `kurtosis` sube junto con energia y `p95_abs_v`, el cambio es mas creible fisicamente.

### Localizacion dentro de la captura

- `event_center_frac`: donde cae el centro energetico dentro de la fila.
- `event_width_frac`: que tan extendida esta la actividad dentro de la fila.

Interpretacion:

- `event_width_frac` bajo significa actividad mas concentrada;
- `event_width_frac` alto significa actividad mas dispersa en la captura;
- si la energia sube y `event_width_frac` baja, la descarga se concentra;
- si ambas suben, la actividad crece y ocupa mas parte de la ventana.

## 5. Como leer combinaciones

No se deben combinar descriptores porque "suena bien", sino porque aportan informacion distinta.

Combinaciones utiles:

- `energy_v2 + p95_abs_v + peak_abs_v`
  - mide severidad global y eventos extremos.
- `active_ratio + pulse_count`
  - mide ocupacion y densidad impulsiva.
- `kurtosis + mean_pulse_amp_v`
  - separa impulsos raros de pulsos realmente energicos.
- `median_dt_s + cv_dt + burstiness`
  - mide estructura temporal interna.
- `energy_v2 + active_ratio + cv_dt`
  - buena combinacion de severidad, ocupacion e irregularidad.

Combinaciones redundantes en este dataset:

- `energy_v2` con `rms_v`
- `pulse_count` con `pulse_rate_hz`
- `energy_v2` con `p95_abs_v` cuando la pregunta es solo "hay mas actividad o no"

Regla practica:

- si dos descriptores cuentan casi la misma historia, se deja uno;
- si uno mide amplitud y otro organizacion temporal, vale la pena mantener ambos.

## 6. Que nos dice `serie_1 (2).mat`

Hallazgos principales del estudio actual:

- bloque principal: filas `497-535`;
- reactivaciones cortas: `537-539`, `552-553`, `556-557`, `561-562`, `565-566`;
- transiciones mas fuertes: filas `503`, `531` y `560`.

Descriptores que mejor describen el bloque dominante:

- `energy_v2`
- `rms_v`
- `p95_abs_v`
- `mean_pulse_amp_v`
- `peak_abs_v`
- `kurtosis`
- `active_ratio`

Descriptores con menor poder de separacion aqui:

- `median_dt_s`
- `iqr_dt_s`
- `event_center_frac`

Lectura tecnica:

- este archivo si contiene una fase de actividad claramente distinta;
- la evidencia mas fuerte no viene de un solo pico, sino de la coincidencia entre amplitud, energia, ocupacion y forma;
- `kurtosis` anade informacion valiosa porque detecta impulsividad extrema fuera del bloque principal;
- los intervalos entre pulsos ayudan, pero no son el eje dominante en este dataset.

## 7. Que tomar en cuenta antes de escribir tesis o paper

- No vender `row_idx` como tiempo fisico absoluto si el experimento no lo documenta asi.
- Hablar de "capturas consecutivas" o "orden de adquisicion" es mas riguroso.
- Si luego identificas en tu cuaderno experimental que en cierto rango de filas se agrego agua con sal, entonces estas figuras pasan de exploratorias a evidenciales.
- Para alarma, la combinacion mas defendible aqui seria `energy_v2 + active_ratio + kurtosis` y, como complemento, `mean_pulse_amp_v`.
- Para un paper de metodo, este dataset demuestra muy bien deteccion de transiciones y bloques activos.
