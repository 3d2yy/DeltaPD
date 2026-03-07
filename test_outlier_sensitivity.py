import os
import shutil
import yaml
from pathlib import Path
import subprocess

def test_sensitivity():
    base_config_path = Path("e:/Carpeta definitiva de Tesis/programas/DeltaPD-main/thesis_campaign/config_material.yaml")
    out_dir = Path("e:/Carpeta definitiva de Tesis/programas/DeltaPD-main/outputs/material_state_outputs")
    
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    thresholds = [0.5, 1.0, 2.0]
    
    for th in thresholds:
        print(f"\\n--- Corriendo sensibilidad con max_valid_dt_s = {th} ---")
        cfg["analysis"]["max_valid_dt_s"] = th
        
        test_cfg_path = Path(f"thesis_campaign/config_material_test_{th}.yaml")
        with open(test_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        cmd = f'python cli.py run-material --config "{test_cfg_path}"'
        # Usamos PYTHONPATH local directamente en el script run
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        subprocess.run(cmd, shell=True, env=env)
        
        # Copiar las figuras clave a una carpeta de sensibilidad
        sens_dir = out_dir / f"sensibilidad_{th}s"
        sens_dir.mkdir(exist_ok=True, parents=True)
        
        for f in out_dir.glob("*.png"):
            shutil.copy(f, sens_dir / f.name)
        
        print(f"Resultados guardados en {sens_dir}")
        test_cfg_path.unlink()

if __name__ == "__main__":
    test_sensitivity()
