from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _notebook_list(root: Path) -> List[Path]:
    nb_dir = root / "notebooks"
    return [
        nb_dir / "00_setup_and_gpu_check.ipynb",
        nb_dir / "01_Setup_and_Data_Audit.ipynb",
        nb_dir / "02_data_and_feature_sets.ipynb",
        nb_dir / "03_SARIMAX_exog.ipynb",
        nb_dir / "04_Prophet_regressors.ipynb",
        nb_dir / "05_GluonTS_DeepAR_exog.ipynb",
        nb_dir / "06_LSTM_global_exog.ipynb",
        nb_dir / "07_Transformer_global_exog.ipynb",
        nb_dir / "08_run_E0_ablation_training.ipynb",
        nb_dir / "09_results_summary_and_plots.ipynb",
        nb_dir / "10_Run_All_Experiments.ipynb",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all notebooks in order using papermill.")
    parser.add_argument("--kernel", type=str, default=None, help="Optional Jupyter kernel name")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder for executed notebooks")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if a notebook fails")
    args = parser.parse_args()

    try:
        import papermill as pm
    except Exception as exc:  # pragma: no cover
        print("papermill is required. Install with: pip install papermill", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    root = _project_root()
    out_dir = Path(args.output_dir) if args.output_dir else (root / "outputs" / "notebook_runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    notebooks = _notebook_list(root)
    for nb in notebooks:
        if not nb.exists():
            print(f"Missing notebook: {nb}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
            continue

        out_path = out_dir / nb.name
        print(f"Running: {nb} -> {out_path}")
        try:
            pm.execute_notebook(
                str(nb),
                str(out_path),
                kernel_name=args.kernel,
                cwd=str(root),
                progress_bar=True,
            )
        except Exception as exc:
            print(f"Failed: {nb} ({exc})", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    print(f"Done. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
