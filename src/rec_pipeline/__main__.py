"""Command-line entry point: python -m rec_pipeline (or python run_pipeline.py from the repository root)."""
import argparse
import sys
import time
import warnings

from . import config as C

STEPS = ["design", "clustering", "operation", "extras", "figures"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="rec_pipeline",
        description="Reproduce the analysis of the paper (dispatch scenarios, financial analysis, machine-learning "
                    "EMS, and operational robustness test).")
    ap.add_argument("--data", choices=["confidential", "synthetic"], default="confidential",
                    help="metered data in data/confidential/ (default) or synthetic stand-in in data/synthetic/")
    ap.add_argument("--steps", default="all",
                    help=f"comma-separated subset of {','.join(STEPS)} (default: all, in this order)")
    ap.add_argument("--make-synthetic", action="store_true",
                    help="(re)generate the synthetic data in data/synthetic/ and exit")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    from .synthetic import generate
    if args.make_synthetic:
        print(f"synthetic data written to {generate()}")
        return 0
    paths = C.run_paths(args.data).ensure()
    if args.data == "synthetic" and not (paths.loads_dir / "community_loads.csv").exists():
        print(f"generating synthetic data in {generate()}")
    steps = STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown steps {unknown}; available: {STEPS}")

    from .data import DataError, load_inputs
    t0 = time.time()
    print(f"data: {args.data} ({paths.loads_dir}) -> results: {paths.results_dir}, figures: {paths.figures_dir}")
    try:
        inp = load_inputs(paths.loads_dir) if any(s != "figures" for s in steps) else None
    except DataError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for s in steps:
        if s == "design":
            from . import design; design.run(inp, paths)
        elif s == "clustering":
            from . import clustering; clustering.run(inp, paths)
        elif s == "operation":
            from . import operation; operation.run(inp, paths)
        elif s == "extras":
            from . import extras; extras.run(inp, paths)
        elif s == "figures":
            from . import figures; figures.run(paths)
    print(f"completed in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
