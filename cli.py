# cli.py
import argparse
from pathlib import Path
from thesis_campaign.thesis_campaign import run_thesis_campaign
from thesis_campaign.material_state import run_material_state
from deltapd.pipeline import run_empirical_pipeline

def main():
    parser = argparse.ArgumentParser(
        prog="deltapd-cli", 
        description="DeltaPD Framework and Thesis Runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Comando 1: Modo Campaña de Tesis
    cmp_parser = subparsers.add_parser(
        "run-campaign", 
        help="Ejecuta la campaña completa de validación desde un archivo YAML"
    )
    cmp_parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("thesis_campaign/config_thesis.yaml"), 
        help="Ruta al YAML de campaña"
    )

    # Comando 2: Modo Estado del Material (Capitulo V)
    mat_parser = subparsers.add_parser(
        "run-material", 
        help="Ejecuta el análisis de la evolución temporal del material (Capítulo V)"
    )
    mat_parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("thesis_campaign/config_material.yaml"), 
        help="Ruta al YAML de estado de material"
    )

    # Comando 3: Validation Pipeline (Legacy Workflow)
    legacy = subparsers.add_parser(
        "run-legacy",
        help="Run the original four-phase synthetic/empirical validation pipeline.",
    )
    legacy.add_argument("-n", "--n-samples", type=int, default=4096)
    legacy.add_argument("--fs", type=float, default=1e9)
    legacy.add_argument("--mc-iterations", type=int, default=100)
    legacy.add_argument("--seed", type=int, default=42)
    legacy.add_argument("-q", "--quiet", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "run-campaign" or args.command == "run-thesis":
        print(f"Iniciando campaña con base en: {args.config}")
        run_thesis_campaign(args.config)
    elif args.command == "run-material":
        print(f"Iniciando análisis de material con base en: {args.config}")
        run_material_state(args.config)
    elif args.command == "run-legacy":
        print("Calling the legacy empirical pipeline...")
        run_empirical_pipeline(
            campaign_dir=Path("./"), 
            fs=args.fs
        )

if __name__ == "__main__":
    main()
