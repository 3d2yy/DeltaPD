import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración Estética (Publication-Ready)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.autolayout': True
})

def find_latest_parquet(directory: Path, prefix: str) -> Path:
    """Busca el archivo parquet más reciente en un directorio según el prefijo."""
    files = list(directory.glob(f"{prefix}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos {prefix}_*.parquet en {directory}")
    # Ordenar por fecha de modificación descendente
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]

def plot_kalman_tracking(df_track: pd.DataFrame, output_dir: Path):
    """
    Gráfico 1: Evolución temporal del intervalo inter-pulso (Delta T)
    con el pronóstico del Filtro de Kalman sobrepuesto.
    """
    print("[FASE 6] Generando Gráfico de Tracking Termodinámico (Kalman)...")
    
    # Extraer arrays
    dt_raw = df_track['delta_t'].values
    dt_kalman = df_track['kalman_filtered'].values
    
    # Simular tiempo continuo acumulado para el eje X (Reconstrucción del Vector Tiempo)
    t_acumulado = np.cumsum(dt_raw)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw scatter (translucido para no saturar)
    ax.scatter(t_acumulado, dt_raw, s=2, alpha=0.3, color='#1f77b4', label=r'Mediciones Crudas ($\Delta t$)')
    
    # Plot Kalman filter (Línea sólida)
    ax.plot(t_acumulado, dt_kalman, linewidth=1.5, color='#ff7f0e', label=r'Filtro de Kalman ($\hat{x}_{k|k}$)')
    
    ax.set_yscale('log')
    ax.set_title('Convergencia del Tracking por Filtro de Kalman 1D', pad=15)
    ax.set_xlabel('Tiempo de Campaña Acumulado (s)')
    ax.set_ylabel(r'Intervalo Inter-Pulso $\Delta t$ (s) [Escala Log]')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    out_path = output_dir / "01_kalman_tracking.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  -> Guardado: {out_path}")

def plot_cusum_alarms(df_track: pd.DataFrame, output_dir: Path):
    """
    Gráfico 2: Evolución de Alarmas Estocásticas CUSUM basadas en el Z-Score.
    """
    print("[FASE 6] Generando Gráfico de Invarianza Estocástica (CUSUM Z-Score)...")
    
    z_scores = df_track['kalman_z_scores'].values
    g_plus = df_track['cusum_g_plus'].values
    g_minus = df_track['cusum_g_minus'].values
    alarm_mask = df_track['cusum_alarms'].values.astype(bool)
    
    # Downsample background traces for performance, but NEVER downsample alarms
    n_points = min(len(z_scores), 5000)
    idx_plot = np.linspace(0, len(z_scores)-1, n_points, dtype=int)
    
    # Full-resolution alarm indices (never lost)
    alarm_indices = np.nonzero(alarm_mask)[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: Z-Scores (Innovaciones estandarizadas)
    ax1.plot(idx_plot, z_scores[idx_plot], linewidth=0.5, color='gray', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.set_title(r'Innovaciones Normalizadas del Filtro de Kalman ($z_k = \nu_k / \sqrt{S_k}$)', pad=10)
    ax1.set_ylabel(r'$z_k$ (adimensional)')
    
    # Pintar alarmas a resolución completa sobre el Z-score
    if len(alarm_indices) > 0:
        for idx in alarm_indices:
            ax1.axvline(idx, color='red', alpha=0.08, linewidth=0.5)
            
    # Subplot 2: CUSUM Accumulators
    ax2.plot(idx_plot, g_plus[idx_plot], linewidth=1.5, color='#2ca02c', label=r'$g^+$ (Deriva Positiva)')
    ax2.plot(idx_plot, g_minus[idx_plot], linewidth=1.5, color='#d62728', label=r'$g^-$ (Deriva Negativa)')
    ax2.axhline(8.0, color='red', linestyle=':', linewidth=2, label=r'Umbral CUSUM $h=8$')
    
    # Alarmas a resolución completa
    if len(alarm_indices) > 0:
        ax2.axvline(alarm_indices[0], color='red', alpha=0.4, linewidth=1, linestyle='--', label=f'Alarmas CUSUM (n={len(alarm_indices)})')
        for idx in alarm_indices[1:]:
            ax2.axvline(idx, color='red', alpha=0.4, linewidth=1, linestyle='--')
            
    ax2.set_title(r'Detección de Cambios de Régimen — CUSUM (drift=$\delta$=0.5, $h$=8)', pad=10)
    ax2.set_xlabel('Índice de Descarga Parcial (k)')
    ax2.set_ylabel(r'Acumulador CUSUM $g$')
    ax2.legend(loc='upper right')
    
    out_path = output_dir / "02_cusum_alarms.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  -> Guardado: {out_path}")

def plot_morphology_distributions(df_morph: pd.DataFrame, output_dir: Path):
    """
    Gráfico 3: Histogramas y KDE físicos (Rise-time y FWHM)
    Para la verificación en la literatura UHF del fenómeno térmico PD.
    """
    print("[FASE 6] Generando Distribuciones Termodinámicas (KDE / Histogramas)...")
    
    tr_ns = df_morph['t_r_ns'].values
    fwhm_ns = df_morph['fwhm_ns'].values
    
    # Filtrar outliers extremos para que el gráfico sea legible (Ej: errores de umbral)
    tr_filtered = tr_ns[(tr_ns > 0) & (tr_ns < np.percentile(tr_ns, 99))]
    fwhm_filtered = fwhm_ns[(fwhm_ns > 0) & (fwhm_ns < np.percentile(fwhm_ns, 99))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma 1: Rise Time
    sns.histplot(tr_filtered, bins=50, kde=True, ax=ax1, color='#9467bd', stat='density')
    ax1.set_title(r'Distribución de Tiempo de Subida ($t_r$)')
    ax1.set_xlabel(r'$t_r$ (ns)')
    ax1.set_ylabel('Densidad de Probabilidad')
    
    # Histograma 2: FWHM
    sns.histplot(fwhm_filtered, bins=50, kde=True, ax=ax2, color='#8c564b', stat='density')
    ax2.set_title('Distribución de Anchura a Media Altura (FWHM)')
    ax2.set_xlabel('FWHM (ns)')
    ax2.set_ylabel('Densidad de Probabilidad')
    
    out_path = output_dir / "03_morphology_distribution.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  -> Guardado: {out_path}")

def main():
    base_dir = Path("e:/SDFDP/DeltaPD")
    export_dir = base_dir / "exports"
    plot_dir = export_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    print("\n[ORQUESTADOR V] Iniciando Ingesta Parquet para Capítulo V")
    
    try:
        # Cargar DataFrames
        track_file = find_latest_parquet(export_dir, "empirical_tracking")
        morph_file = find_latest_parquet(export_dir, "empirical_morphology")
        
        print(f"  -> Tracking Parquet: {track_file.name}")
        print(f"  -> Morphology Parquet: {morph_file.name}")
        
        df_track = pd.read_parquet(track_file)
        df_morph = pd.read_parquet(morph_file)
        
        print(f"  -> Filas Tracking: {len(df_track):,} | Filas Morfología: {len(df_morph):,}")
        
    except Exception as e:
        print(f"Error crítico al cargar repositorios Parquet: {e}")
        return
        
    print("\n[INICIANDO RENDERIZACIÓN DE ALTA DENSIDAD]")
    plot_kalman_tracking(df_track, plot_dir)
    plot_cusum_alarms(df_track, plot_dir)
    plot_morphology_distributions(df_morph, plot_dir)
    
    print("\n[ESTADO] Extracción del Capítulo V Completada - Gráficos Exportados.")

if __name__ == "__main__":
    main()
