# Esquema de Capitulos IV y V

## Proposito

Este documento fija una estructura de trabajo para los Capitulos IV y V de la tesis, con una nomenclatura experimental unica y una tabla maestra comun para todos los analisis.

La idea es separar con claridad:

- **Capitulo IV**: repetibilidad, robustez y comparacion entre antenas.
- **Capitulo V**: metodo integrado basado en `delta_t + pseudo-PRPD ciego`.

## Convencion experimental oficial

### Grupo A. Pruebas principales

Estas son las tres pruebas base del programa:

| Dataset | Etiqueta de tesis | Canal | Antena |
|---|---|---|---|
| P1 | Prueba 1 - Internas | CH2 | Deepace |
| P1 | Prueba 1 - Internas | CH3 | Vivaldi antipodal propuesta |
| P1 | Prueba 1 - Internas | CH4 | Bioinspirada |
| P2 | Prueba 2 - Superficiales | CH2 | Deepace |
| P2 | Prueba 2 - Superficiales | CH3 | Vivaldi antipodal propuesta |
| P2 | Prueba 2 - Superficiales | CH4 | Bioinspirada |
| P3 | Prueba 3 - Ensayo de Fuentes Multiples Simultaneas | CH2 | Deepace |
| P3 | Prueba 3 - Ensayo de Fuentes Multiples Simultaneas | CH3 | Vivaldi antipodal propuesta |
| P3 | Prueba 3 - Ensayo de Fuentes Multiples Simultaneas | CH4 | Bioinspirada |

### Grupo B. Pruebas gemelas

Estas son las tres pruebas de repetibilidad con antena gemela:

| Dataset | Etiqueta de tesis | Canal | Antena |
|---|---|---|---|
| G1 | Prueba 1 - Internas (Gemelas) | CH2 | Gemela Vivaldi antipodal propuesta |
| G1 | Prueba 1 - Internas (Gemelas) | CH3 | Vivaldi antipodal propuesta |
| G1 | Prueba 1 - Internas (Gemelas) | CH4 | Bioinspirada |
| G2 | Prueba 2 - Superficiales (Gemelas) | CH2 | Gemela Vivaldi antipodal propuesta |
| G2 | Prueba 2 - Superficiales (Gemelas) | CH3 | Vivaldi antipodal propuesta |
| G2 | Prueba 2 - Superficiales (Gemelas) | CH4 | Bioinspirada |
| G3 | Prueba 3 - Ensayo de Fuentes Multiples Simultaneas (Gemelas) | CH2 | Gemela Vivaldi antipodal propuesta |
| G3 | Prueba 3 - Ensayo de Fuentes Multiples Simultaneas (Gemelas) | CH3 | Vivaldi antipodal propuesta |
| G3 | Prueba 3 - Ensayo de Fuentes Multiples Simultaneas (Gemelas) | CH4 | Bioinspirada |

### Regla de nombres para tablas y figuras

Para evitar ambiguedades, se recomienda usar estos nombres canonicamente en tablas y figuras:

| Canal | Benchmark | Gemelas |
|---|---|---|
| CH2 | Deepace | Gemela Vivaldi antipodal propuesta |
| CH3 | Vivaldi antipodal propuesta | Vivaldi antipodal propuesta |
| CH4 | Bioinspirada | Bioinspirada |

## Capitulo IV

### Objetivo

Demostrar que el sistema de adquisicion y las antenas producen mediciones repetibles, comparables y robustas bajo escenarios equivalentes.

### 4.1 Montaje experimental

- Describir las tres antenas evaluadas por grupo experimental.
- Separar claramente el grupo principal `P1-P3` del grupo de repetibilidad `G1-G3`.
- Dejar explicito que el foco del capitulo es la calidad experimental del sistema y no el diagnostico final del material.

### 4.2 Flujo de procesamiento

Base de codigo:

- [loader.py](src/deltapd/loader.py)
- [descriptors.py](src/deltapd/descriptors.py)
- [blind_prpd.py](src/deltapd/blind_prpd.py)
- [evaluate_gemelas_repeatability.py](scripts/evaluate_gemelas_repeatability.py)

Pasos:

1. Carga de senal por canal.
2. Deteccion de pulsos.
3. Estimacion ciega de frecuencia.
4. Reconstruccion pseudo-PRPD.
5. Extraccion de metricas de repetibilidad.

### 4.3 Metricas del Capitulo IV

Estas metricas deben salir de una tabla unica y compartida:

- `pulse_count`
- `blind_freq_hz`
- `mean_peak_v`
- `std_peak_v`
- `phase_entropy_global`
- `phase_width_pos_deg`
- `phase_width_neg_deg`
- `inlier_ratio`

### 4.4 Resultados de repetibilidad

Resultados ya disponibles:

- [gemelas_repeatability_metrics.csv](outputs/gemelas_repeatability/gemelas_repeatability_metrics.csv)
- [gemelas_repeatability_differences.csv](outputs/gemelas_repeatability/gemelas_repeatability_differences.csv)
- [g1_gemelas_prpd_comparison.png](outputs/gemelas_repeatability/g1_gemelas_prpd_comparison.png)
- [g2_gemelas_prpd_comparison.png](outputs/gemelas_repeatability/g2_gemelas_prpd_comparison.png)
- [g3_gemelas_prpd_comparison.png](outputs/gemelas_repeatability/g3_gemelas_prpd_comparison.png)

Mensajes que debe sostener este bloque:

- `G1` y `G2` deben usarse como evidencia principal de repetibilidad.
- `G3` debe usarse como escenario de mayor complejidad.
- La `Bioinspirada` entra como contraste de desempeno, no como simple control pasivo.

### 4.5 Robustez frente a parametros

Usar como evidencia:

- [blind_prpd_threshold_sensitivity.csv](outputs/blind_prpd_frequency_phase_eval/blind_prpd_threshold_sensitivity.csv)
- [blind_prpd_threshold_sensitivity.png](outputs/blind_prpd_frequency_phase_eval/blind_prpd_threshold_sensitivity.png)

Mensaje:

- La reconstruccion pseudo-PRPD y la frecuencia ciega son estables en una banda razonable de umbral.
- La zona operativa recomendada, hasta ahora, es `4.5-5.5 sigma`.

### 4.6 Cierre del Capitulo IV

La conclusion del capitulo debe responder tres preguntas:

1. El sistema mide de forma repetible.
2. La metodologia es robusta a cambios razonables de parametrizacion.
3. La plataforma es suficientemente estable para soportar el analisis metodologico del Capitulo V.

## Capitulo V

### Objetivo

Demostrar que la combinacion de `delta_t` y pseudo-PRPD ciego permite extraer estructura temporal y fasica util para caracterizar la actividad de descargas parciales UHF.

### 5.1 Hipotesis central

Un pipeline integrado basado en `delta_t` y pseudo-PRPD ciego permite detectar y caracterizar cambios de regimen en actividad PD UHF con mayor riqueza descriptiva que usar solo conteo de pulsos o amplitud.

### 5.2 Pipeline metodologico

Base de codigo:

- [run_integrated_pd_experiment.py](scripts/run_integrated_pd_experiment.py)
- [blind_prpd.py](src/deltapd/blind_prpd.py)
- [statistics.py](src/deltapd/statistics.py)
- [trackers.py](src/deltapd/trackers.py)
- [evaluate_blind_prpd_frequency_phase.py](scripts/evaluate_blind_prpd_frequency_phase.py)
- [compare_blind_prpd_variants.py](scripts/compare_blind_prpd_variants.py)

Pasos:

1. Deteccion de pulsos.
2. Construccion de `delta_t`.
3. Estimacion ciega de frecuencia.
4. Reconstruccion pseudo-PRPD.
5. Extraccion de metricas temporales.
6. Extraccion de metricas fasicas.
7. Comparacion entre `P1`, `P2` y `P3`.

### 5.3 Metricas del Capitulo V

Temporales:

- `median_dt_s`
- `iqr_dt_s`
- `cv_dt`
- `burstiness_mean`
- `fano_global`

Fasicas:

- `blind_freq_hz`
- `phase_entropy_global`
- `phase_width_pos_deg`
- `phase_width_neg_deg`
- `inlier_ratio`
- `mean_peak_v`

### 5.4 Resultados integrados

Resultados ya disponibles:

- [integrated_pd_metrics_p1_p2_p3.csv](outputs/integrated_pd_experiment/integrated_pd_metrics_p1_p2_p3.csv)
- [integrated_pd_prpd_comparison_p1_p2_p3.png](outputs/integrated_pd_experiment/integrated_pd_prpd_comparison_p1_p2_p3.png)
- [blind_prpd_method_metrics_p1_p2_p3.csv](outputs/blind_prpd_variants_all/blind_prpd_method_metrics_p1_p2_p3.csv)
- [p1_frequency_phase_curve.png](outputs/blind_prpd_frequency_phase_eval/p1_frequency_phase_curve.png)
- [p2_frequency_phase_curve.png](outputs/blind_prpd_frequency_phase_eval/p2_frequency_phase_curve.png)
- [p3_frequency_phase_curve.png](outputs/blind_prpd_frequency_phase_eval/p3_frequency_phase_curve.png)

Lectura esperada:

- `P2` es el caso fuerte del metodo.
- `P1` es el caso intermedio.
- `P3` es el caso limite o multi-fuente.

### 5.5 Validacion del PRPD ciego

Mensaje principal:

- La mejor configuracion validada hasta ahora es `calibracion global ciega + KDE ponderado por amplitud`.
- La interpretacion debe ser cualitativa y complementaria.
- No se debe vender como sustituto directo de un PRPD IEC 60270 referenciado.

### 5.6 Relacion con la literatura

La redaccion recomendada es:

- `compatible con`
- `concuerda cualitativamente con`
- `reproduce morfologias esperadas`

No usar como afirmacion fuerte:

- `identifica definitivamente`
- `clasifica de manera concluyente`

### 5.7 Limitaciones

- No es PRPD convencional con referencia AC directa.
- La interpretacion por tipo de descarga sigue siendo inferencial.
- Los escenarios multi-fuente degradan la estabilidad local.
- El clasificador actual no debe ser eje del capitulo.

### 5.8 Cierre del Capitulo V

La conclusion del capitulo debe responder:

1. El pseudo-PRPD ciego aporta estructura fasica util.
2. `delta_t` aporta estructura temporal complementaria.
3. Ambos, integrados, forman una metodologia reproducible y defendible.

## Tabla maestra de tesis

### Definicion

La tabla maestra debe contener una fila por combinacion `dataset x canal`, es decir:

- `P1-P3` con `CH2-CH4`
- `G1-G3` con `CH2-CH4`

Total esperado:

- `18` filas para la tabla resumen principal

### Esquema de columnas

| Columna | Tipo | Uso | Capitulo |
|---|---|---|---|
| `group_family` | texto | `benchmark` o `gemelas` | IV y V |
| `dataset_key` | texto | `P1`, `P2`, `P3`, `G1`, `G2`, `G3` | IV y V |
| `dataset_label` | texto | nombre largo de la prueba | IV y V |
| `channel` | texto | `CH2`, `CH3`, `CH4` | IV y V |
| `antenna_label` | texto | nombre oficial de antena | IV y V |
| `source_file` | texto | ruta al CSV fuente | IV y V |
| `fs_hz` | numerico | frecuencia de muestreo | IV y V |
| `threshold_sigma` | numerico | umbral de deteccion usado | IV y V |
| `min_separation_s` | numerico | separacion minima entre pulsos | IV y V |
| `pulse_count` | numerico | pulsos detectados | IV y V |
| `blind_freq_hz` | numerico | frecuencia ciega estimada | IV y V |
| `mean_peak_v` | numerico | amplitud media de pico | IV y V |
| `std_peak_v` | numerico | dispersion de amplitud | IV y V |
| `phase_entropy_global` | numerico | dispersion fasica global | IV y V |
| `phase_width_pos_deg` | numerico | ancho de fase semiciclo positivo | IV y V |
| `phase_width_neg_deg` | numerico | ancho de fase semiciclo negativo | IV y V |
| `inlier_ratio` | numerico | proporcion de puntos retenidos | IV y V |
| `median_dt_s` | numerico | mediana de `delta_t` | V |
| `iqr_dt_s` | numerico | rango intercuartil de `delta_t` | V |
| `cv_dt` | numerico | coeficiente de variacion de `delta_t` | V |
| `burstiness_mean` | numerico | indice medio de burstiness | V |
| `fano_global` | numerico | factor de Fano | V |
| `notes` | texto | observaciones o flags | IV y V |

### Tabla maestra recomendada para repetibilidad por pares

Ademas de la tabla principal, conviene una tabla secundaria para comparaciones por pares dentro de cada prueba:

| Columna | Tipo | Uso |
|---|---|---|
| `dataset_key` | texto | `G1`, `G2`, `G3` |
| `dataset_label` | texto | nombre largo |
| `antenna_a` | texto | antena A |
| `antenna_b` | texto | antena B |
| `freq_diff_hz` | numerico | diferencia de frecuencia ciega |
| `mean_peak_diff_v` | numerico | diferencia de amplitud media |
| `entropy_diff` | numerico | diferencia de entropia |
| `phase_width_pos_diff_deg` | numerico | diferencia de ancho positivo |
| `phase_width_neg_diff_deg` | numerico | diferencia de ancho negativo |
| `pulse_ratio` | numerico | razon de conteo de pulsos |

Esta tabla ya tiene un antecedente en:

- [gemelas_repeatability_differences.csv](outputs/gemelas_repeatability/gemelas_repeatability_differences.csv)

## Figuras clave a conservar

### Para Capitulo IV

- [g1_gemelas_prpd_comparison.png](outputs/gemelas_repeatability/g1_gemelas_prpd_comparison.png)
- [g2_gemelas_prpd_comparison.png](outputs/gemelas_repeatability/g2_gemelas_prpd_comparison.png)
- [g3_gemelas_prpd_comparison.png](outputs/gemelas_repeatability/g3_gemelas_prpd_comparison.png)
- [blind_prpd_threshold_sensitivity.png](outputs/blind_prpd_frequency_phase_eval/blind_prpd_threshold_sensitivity.png)

### Para Capitulo V

- [integrated_pd_prpd_comparison_p1_p2_p3.png](outputs/integrated_pd_experiment/integrated_pd_prpd_comparison_p1_p2_p3.png)
- [p1_frequency_phase_curve.png](outputs/blind_prpd_frequency_phase_eval/p1_frequency_phase_curve.png)
- [p2_frequency_phase_curve.png](outputs/blind_prpd_frequency_phase_eval/p2_frequency_phase_curve.png)
- [p3_frequency_phase_curve.png](outputs/blind_prpd_frequency_phase_eval/p3_frequency_phase_curve.png)

## Estado actual de alineacion

Ya quedaron alineados:

1. La configuracion de tesis en [campaign/config_thesis.yaml](campaign/config_thesis.yaml) y [thesis_campaign/config_thesis.yaml](thesis_campaign/config_thesis.yaml) ahora refleja explicitamente `P1-P3` y `G1-G3`.
2. La tabla maestra completa para `CH2-CH4` ya fue exportada en [thesis_master_metrics.csv](outputs/thesis_master/thesis_master_metrics.csv).
3. La tabla secundaria de comparaciones por pares ya fue exportada en [thesis_master_pairwise_differences.csv](outputs/thesis_master/thesis_master_pairwise_differences.csv).

Queda pendiente:

1. Usar la tabla maestra como fuente unica para captions, tablas y redaccion de los capitulos.
2. Decidir si los scripts exploratorios viejos se conservan como soporte o se retiran del flujo principal.

## Decision metodologica congelada hasta ahora

- Deteccion basada en pulsos sobre la senal UHF cargada desde CSV.
- Estimacion de frecuencia por calibracion global ciega.
- Reconstruccion pseudo-PRPD con limpieza de outliers.
- Visualizacion preferida: `KDE` ponderado por amplitud.
- Zona de trabajo recomendada para umbral: `4.5-5.5 sigma`.
