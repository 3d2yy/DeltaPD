import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _add_stage_shading(ax, df: pd.DataFrame, is_time: bool = False):
    """Agrega sombreado o líneas verticales donde cambia la etapa (si existe `stage`)."""
    if "stage" not in df.columns:
        return
    
    stages = df["stage"].to_numpy()
    changes_idx = np.where(stages[:-1] != stages[1:])[0]
    
    colors = ['#f9f9f9', '#ececec']
    
    if is_time:
        x_vals = df["toa_s"].to_numpy()
        current_x = x_vals[0] if len(x_vals) > 0 else 0
        end_x = x_vals[-1] if len(x_vals) > 0 else 0
        
        color_idx = 0
        for change_idx in changes_idx:
            change_x = x_vals[change_idx]
            ax.axvspan(current_x, change_x, facecolor=colors[color_idx % 2], alpha=0.5, zorder=-1)
            ax.axvline(x=change_x, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
            current_x = change_x
            color_idx += 1
            
        ax.axvspan(current_x, end_x, facecolor=colors[color_idx % 2], alpha=0.5, zorder=-1)
    else:
        current_idx = 0
        color_idx = 0
        for change_idx in changes_idx:
            ax.axvspan(current_idx, change_idx, facecolor=colors[color_idx % 2], alpha=0.5, zorder=-1)
            ax.axvline(x=change_idx, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
            current_idx = change_idx
            color_idx += 1
            
        ax.axvspan(current_idx, len(df), facecolor=colors[color_idx % 2], alpha=0.5, zorder=-1)

def plot_raw_with_detections(t: np.ndarray, x: np.ndarray, toa: np.ndarray, out_png: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [1, 1]})
    
    # Global View
    ax1.plot(t * 1e6, x, color="gray", alpha=0.7, label="Señal")
    for tp in toa:
        ax1.axvline(tp * 1e6, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.set_xlabel("Tiempo (µs)")
    ax1.set_ylabel("Voltaje")
    ax1.set_title("VISTA GLOBAL: Señal Cruda con Detecciones")
    
    # Zoom View (Focusing on the first 5 pulses or a 10µs window around the first detection)
    if len(toa) > 0:
        first_pulse_us = toa[0] * 1e6
        zoom_start = max(0, first_pulse_us - 2.0)
        zoom_end = first_pulse_us + 8.0  # 10us window
        
        # Mask the data for plotting speed in zoom
        mask = (t * 1e6 >= zoom_start) & (t * 1e6 <= zoom_end)
        ax2.plot(t[mask] * 1e6, x[mask], color="blue", alpha=0.8, label="Señal Zoom")
        
        for tp in toa:
            if zoom_start <= tp * 1e6 <= zoom_end:
                ax2.axvline(tp * 1e6, color="red", linestyle="--", alpha=0.8, linewidth=1.5)
        
        ax2.set_xlim(zoom_start, zoom_end)
        ax2.set_xlabel("Tiempo (µs)")
        ax2.set_ylabel("Voltaje")
        ax2.set_title("VISTA DETALLE (10 µs): Señal Magnificada")
    else:
        ax2.text(0.5, 0.5, "Sin detecciones", ha='center', va='center')
        ax2.set_axis_off()
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_delta_t_series(df_delta: pd.DataFrame, out_png: str, is_log: bool = False):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    
    col = "log10_dt" if is_log else "delta_t_s"
    ylabel = "log10(Δt) (s)" if is_log else "Δt (s)"
    title_suffix = "(Log10)" if is_log else "(Lineal)"
    
    ax.plot(df_delta["toa_s"], df_delta[col], linewidth=1.2)
    if is_log:
        ax.set_ylim([-5, 0.5])
        
    _add_stage_shading(ax, df_delta, is_time=True)
    
    ax.set_xlabel("Tiempo real del ensayo (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Serie temporal de Δt {title_suffix}")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_delta_t_histogram(df_delta: pd.DataFrame, out_png: str, is_log: bool = False):
    plt.figure(figsize=(7.5, 4.8))
    
    col = "log10_dt" if is_log else "delta_t_s"
    xlabel = "log10(Δt) (s)" if is_log else "Δt (s)"
    title_suffix = "(Log10)" if is_log else "(Lineal)"
    
    plt.hist(df_delta[col], bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de Δt {title_suffix}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_rate_series(df_delta: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    
    ax.plot(df_delta["toa_s"], df_delta["pulse_rate_hz"], color="lightblue", alpha=0.6, linewidth=1.0, label="Tasa instantánea")
    if "rolling_rate_hz" in df_delta.columns:
        ax.plot(df_delta["toa_s"], df_delta["rolling_rate_hz"], color="darkblue", linewidth=1.8, label="Tasa suavizada")
        
    _add_stage_shading(ax, df_delta, is_time=True)
    
    ax.set_xlabel("Tiempo real del ensayo (s)")
    ax.set_ylabel("Tasa de repetición (Hz)")
    ax.set_title("Tasa de eventos derivada de Δt")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_rolling_stats(df_delta: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    
    ax.plot(df_delta["toa_s"], df_delta["rolling_median_dt"], linewidth=1.8, label="Mediana móvil")
    
    w = len(df_delta) // 10 if len(df_delta) > 50 else 5
    q1_exact = df_delta["delta_t_s"].rolling(25, min_periods=5).quantile(0.25)
    q3_exact = df_delta["delta_t_s"].rolling(25, min_periods=5).quantile(0.75)
    
    ax.fill_between(df_delta["toa_s"], q1_exact, q3_exact, alpha=0.25, label="IQR móvil")
    _add_stage_shading(ax, df_delta, is_time=True)
    
    ax.set_xlabel("Tiempo real del ensayo (s)")
    ax.set_ylabel("Δt (s)")
    ax.set_title("Evolución local de Δt")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_ewma_cusum(df_delta: pd.DataFrame, alpha: float, cusum_k: float, cusum_h: float, out_png: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Corremos CUSUM sobre log10_dt para evitar la desestabilización por cola larga de Δt
    x = df_delta["log10_dt"].to_numpy(dtype=float)

    ewma = np.zeros_like(x)
    warmup_n = min(50, len(x))
    if warmup_n > 0:
        ewma[0] = np.median(x[:warmup_n])
    else:
        ewma[0] = x[0] if len(x) > 0 else 0
        
    for i in range(1, len(x)):
        ewma[i] = alpha * x[i] + (1 - alpha) * ewma[i-1]

    mu = np.mean(x)
    sigma = np.std(x) if np.std(x) > 0 else 1.0
    z = (x - mu) / sigma

    cp = np.zeros_like(z)
    cn = np.zeros_like(z)
    for i in range(1, len(z)):
        cp[i] = max(0, cp[i-1] + z[i] - cusum_k)
        cn[i] = min(0, cn[i-1] + z[i] + cusum_k)

    if "toa_s" in df_delta.columns:
        t_vals = df_delta["toa_s"].to_numpy()
    else:
        t_vals = np.arange(len(ewma))

    ax.plot(t_vals, ewma, linewidth=1.8, label="EWMA de log10(Δt)")
    ax.plot(t_vals, cp, linewidth=1.2, label="CUSUM+")
    ax.plot(t_vals, cn, linewidth=1.2, label="CUSUM-")
    ax.axhline(cusum_h, linestyle="--", color="black", linewidth=1.0)
    ax.axhline(-cusum_h, linestyle="--", color="black", linewidth=1.0)
    
    _add_stage_shading(ax, df_delta, is_time=True)
    
    ax.set_xlabel("Tiempo real del ensayo (s)")
    ax.set_ylabel("Valor procesado (Z-Score sobre Log)")
    ax.set_title("Seguimiento EWMA/CUSUM Robusto (sobre Log10(Δt))")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_advanced_analytics(df_delta: pd.DataFrame, out_png: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    t_vals = df_delta["toa_s"] if "toa_s" in df_delta.columns else df_delta["event_idx"]
    xlabel = "Tiempo real del ensayo (s)" if "toa_s" in df_delta.columns else "Índice de evento"
    
    # Weibull Beta
    ax1.plot(t_vals, df_delta["rolling_weibull_beta"], color="purple", linewidth=1.5, label=r"Weibull Shape ($\beta$)")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.8, label="Poisson (Random)")
    _add_stage_shading(ax1, df_delta, is_time=("toa_s" in df_delta.columns))
    ax1.set_ylabel(r"Forma $\beta$")
    ax1.set_title("Evolución de Parámetros Estadísticos Avanzados (Q1)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left")
    
    # Burstiness Index
    ax2.plot(t_vals, df_delta["rolling_burstiness"], color="darkorange", linewidth=1.5, label="Burstiness Index (B)")
    ax2.axhline(0.0, color="gray", linestyle="--", alpha=0.8, label="Neutral (Poisson)")
    ax2.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="High Burst")
    _add_stage_shading(ax2, df_delta, is_time=("toa_s" in df_delta.columns))
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Índice B [-1, 1]")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_blind_prpd(df_delta: pd.DataFrame, out_png: str):
    if "prpd_phase_deg" not in df_delta.columns or "peak_v" not in df_delta.columns:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = df_delta["prpd_phase_deg"].to_numpy()
    y = df_delta["peak_v"].to_numpy()
    
    # 2D Histogram for PRPD density scatter
    h = ax.hist2d(x, y, bins=[360, 100], cmap="turbo", range=[[0, 360], [0, np.max(y)*1.1]], cmin=1)
    fig.colorbar(h[3], ax=ax, label="Conteos")
    
    # Overlay a sine wave to guide the eye
    t_sin = np.linspace(0, 360, 360)
    y_sin = np.max(y) * 0.5 * np.sin(np.radians(t_sin)) + (np.max(y) / 2)
    ax.plot(t_sin, y_sin, color="white", alpha=0.4, linestyle="--", label="Ref. Fase AC (50Hz)")

    ax.set_xlim(0, 360)
    ax.set_xlabel("Fase Ciega Reconstruida (Grados)")
    ax.set_ylabel("Amplitud Peak (V)")
    ax.set_title("Phase-Resolved Partial Discharge (PRPD) - Blind Sync @ 50Hz")
    
    # Add textual note about blind phase reconstruction
    ax.text(0.5, -0.16, 
            "Nota: La fase se reconstruye desde el tiempo de llegada del evento asumiendo sincronía a 50 Hz;\nno existe referencia de fase AC medida.", 
            horizontalalignment='center', verticalalignment='top', 
            transform=ax.transAxes, fontsize=9, color='gray', style='italic')

    ax.grid(True, linestyle=":", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
