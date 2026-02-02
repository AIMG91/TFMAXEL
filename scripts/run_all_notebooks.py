from __future__ import annotations

import argparse
import os
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


def _resolve_kernel_name(requested: str | None) -> str | None:
    """Resolve a user-provided kernel string to an installed kernelspec name.

    Users often confuse the kernelspec *name* (what nbclient expects) with the
    kernel *display name* shown in UI. This helper tries:
      1) exact match by kernelspec name
      2) exact match by display name
      3) case-insensitive substring match on display name
    """

    if requested is None:
        return None

    requested = requested.strip()
    if not requested:
        return None

    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except Exception:
        # If jupyter_client isn't available, we can't resolve; return as-is.
        return requested

    ksm = KernelSpecManager()
    specs = ksm.find_kernel_specs()  # name -> resource_dir

    # 1) exact kernelspec name
    if requested in specs:
        return requested

    # Build display map
    display_by_name: dict[str, str] = {}
    for name in specs.keys():
        try:
            display_by_name[name] = ksm.get_kernel_spec(name).display_name
        except Exception:
            display_by_name[name] = name

    # 2) exact display name
    for name, display in display_by_name.items():
        if display == requested:
            return name

    # 3) substring display-name match (case-insensitive)
    requested_lower = requested.lower()
    for name, display in display_by_name.items():
        if requested_lower in (display or "").lower():
            return name

    # Not found: print a helpful list
    print("Available kernels (kernelspec name -> display name):", file=sys.stderr)
    for name in sorted(display_by_name.keys()):
        print(f"- {name} -> {display_by_name[name]}", file=sys.stderr)

    return requested


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all notebooks in order. Uses papermill when available; "
            "falls back to nbclient if papermill is not installed."
        )
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="python3",
        help="Jupyter kernel name (default: python3)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder for executed notebooks")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if a notebook fails")
    args = parser.parse_args()

    kernel_name = _resolve_kernel_name(args.kernel)

    def run_notebook(nb_path: Path, out_path: Path) -> None:
        try:
            import papermill as pm
        except Exception as import_exc:
            pm = None
            print(f"Papermill not available ({import_exc}); falling back to nbclient.")

        if pm is not None:
            print("Executor: papermill")

            pm_kwargs = {
                "cwd": str(exec_cwd),
                "progress_bar": True,
            }
            if kernel_name:
                pm_kwargs["kernel_name"] = kernel_name

            # If the notebook fails, propagate the real notebook error.
            pm.execute_notebook(str(nb_path), str(out_path), **pm_kwargs)
            return

        try:
            import nbformat
            from nbclient import NotebookClient

            print("Executor: nbclient")
            nb = nbformat.read(str(nb_path), as_version=4)

            nbclient_kwargs = {
                "timeout": None,
                "allow_errors": args.continue_on_error,
            }
            if kernel_name:
                nbclient_kwargs["kernel_name"] = kernel_name

            client = NotebookClient(nb, **nbclient_kwargs)

            prev_cwd = os.getcwd()
            try:
                os.chdir(exec_cwd)
                client.execute()
            finally:
                os.chdir(prev_cwd)

            nbformat.write(nb, str(out_path))
        except Exception as exc2:
            print("Notebook execution failed using nbclient.", file=sys.stderr)
            print(f"Execution error: {exc2}", file=sys.stderr)
            raise

    root = _project_root()
    exec_cwd = root / "notebooks"
    if not exec_cwd.exists():
        exec_cwd = root
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
            run_notebook(nb, out_path)
        except Exception as exc:
            print(f"Failed: {nb} ({exc})", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    print(f"Done. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
