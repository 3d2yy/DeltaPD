import numpy as np
from pathlib import Path
import tempfile
import pytest

from deltapd.loader import load_empirical_signal

def test_load_empirical_signal_real_fixture():
    """
    Prueba que la función load_empirical_signal maneje correctamente
    los bordes y las cabeceras comunes de un osciloscopio real (Rigol, Tektronix).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "CH3_real_min.csv"
        
        # Simula el formato de un csv real exportado por osciloscopio
        lines = [
            "Model,DS1054Z",
            "CH3,2.00V,0.00V",
            "SampleRate,1.000000e+09",
            "X,,,",
            "Time,CH3",
            "-5.000000e-06,0.04",
            "-4.999000e-06,0.08",
            "-4.998000e-06,-0.02",
            "-4.997000e-06,0.00",
            "-4.996000e-06,0.02",
        ]
        
        csv_path.write_text("\n".join(lines), encoding="utf-8")
        
        # Validar ingesta
        x, fs = load_empirical_signal(str(csv_path), preserve_amplitude=True)
        
        # Asumiendo que detecta correctamente el encabezado 'Time' o lee el final numérico
        assert len(x) > 0, "No se leyó ninguna muestra del CSV"
        # La señal es centrada (- DC offset), el mean_original = 0.024, así que x[0] = 0.04 - 0.024 = 0.016
        assert np.isclose(x[0], 0.016), f"Se leyó {x[0]} en lugar de 0.016"
        assert len(x) == 5, "Se debieron leer 5 muestras"
        
        # Validar consistencia de FS con la diferencia de tiempos en el CSV
        # 1ns de separación = 1e9 Hz
        expected_fs = 1.0 / (1e-9)
        assert np.isclose(fs, expected_fs, rtol=1e-3), f"Fs detectado {fs} != {expected_fs}"
