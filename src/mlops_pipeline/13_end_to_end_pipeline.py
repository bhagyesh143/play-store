import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "01_load_data.py",
    "02_validate_data.py",
    "03_preprocessing.py",
    "04_validate_outputs.py",
    "05_train_registry.py",
    "06_model_evaluation.py",
    "07_quality_gate.py",
    "08_automate_lifecycle.py",
    "09_inference.py",
    "10_failure_injection.py",
    "11_prepare_deployment.py",
]


def main():
    print("=" * 70)
    print("GOOGLE PLAY STORE - END-TO-END MLOPS PIPELINE (LABS 1-13)")
    print("=" * 70)

    for script in SCRIPTS:
        script_path = ROOT / "src" / "mlops_pipeline" / script
        if not script_path.exists():
            print(f"\n[ERROR] Script not found: {script_path}")
            sys.exit(1)

        print(f"\n{'=' * 70}")
        print(f"Running Stage: {script}")
        print("=" * 70)

        result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[FAILED] Pipeline stopped at {script} (code {result.returncode})")
            sys.exit(result.returncode)

        print(f"[SUCCESS] {script} passed.")

    print("\n" + "=" * 70)
    print("END-TO-END MLOPS PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
