from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import json


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
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Start execution from this notebook filename (e.g., 06_LSTM_global_exog.ipynb)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only the given notebook filename (can be passed multiple times)",
    )
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

    run_summary_path = out_dir / "_run_summary.json"
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "kernel_requested": args.kernel,
        "kernel_resolved": kernel_name,
        "exec_cwd": str(exec_cwd),
        "output_dir": str(out_dir),
        "notebooks": [],
    }

    notebooks = _notebook_list(root)

    if args.only:
        only_set = {n.strip() for n in args.only if n and n.strip()}
        notebooks = [nb for nb in notebooks if nb.name in only_set]

    if args.start_from:
        start_name = args.start_from.strip()
        if start_name:
            try:
                start_idx = [nb.name for nb in notebooks].index(start_name)
                notebooks = notebooks[start_idx:]
            except ValueError:
                print(f"--start-from notebook not found in list: {start_name}", file=sys.stderr)
                print("Available notebooks:", file=sys.stderr)
                for nb in notebooks:
                    print(f"- {nb.name}", file=sys.stderr)
                return 2

    failures: list[dict] = []
    for nb in notebooks:
        if not nb.exists():
            print(f"Missing notebook: {nb}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
            continue

        out_path = out_dir / nb.name
        print(f"Running: {nb} -> {out_path}")
        nb_rec = {
            "notebook": str(nb),
            "output": str(out_path),
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }
        try:
            run_notebook(nb, out_path)
            nb_rec["status"] = "ok"
        except Exception as exc:
            nb_rec["status"] = "failed"
            nb_rec["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(nb_rec)

            print(f"Failed: {nb} ({exc})", file=sys.stderr)
            traceback.print_exc()

            # Write an updated summary eagerly so the last failure is persisted.
            try:
                summary["notebooks"].append(nb_rec)
                summary["failures"] = failures
                summary["updated_at"] = datetime.now(timezone.utc).isoformat()
                run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except Exception:
                pass

            if not args.continue_on_error:
                summary["notebooks"].append(nb_rec)
                summary["failures"] = failures
                summary["finished_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return 1

        summary["notebooks"].append(nb_rec)

    summary["failures"] = failures
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass

    if failures:
        print("Done with failures:", file=sys.stderr)
        for f in failures:
            print(f"- {Path(f['notebook']).name}: {f['error']}", file=sys.stderr)
        print(f"Summary written to: {run_summary_path}", file=sys.stderr)
        return 1

    print(f"Done. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
