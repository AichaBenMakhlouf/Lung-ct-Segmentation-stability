"""Run the gradient-accumulation experiments of the article.

    baseline      baseline configuration, gradient accumulation   20 seeds        (Section 4.3)
    progressive   progressive protocol, arms A-D                  4 arms x 12     (Section 4.4)
    benchmark     architecture benchmark                          4 archs x 5     (Section 4.5)
    splits        split sensitivity, U-Net and ResUNet++          random 6 seeds, (Section 4.6)
                                                                  Z-blocked 6 x 4 folds,
                                                                  leave-one-patient-out 10 folds

The literal-batch-10 baseline runs separately (scripts/train_literal_batch10.py).
Results are written to results/runs/. Finished runs are skipped, so the command can be
re-launched to resume.

Usage (from the repository root):
    python scripts/run_experiments.py                      # everything
    python scripts/run_experiments.py --only benchmark     # one experiment
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lungct import experiments, paths  # noqa: E402

STEPS = {
    "baseline": lambda d: experiments.run_baseline_accumulation(n_seeds=20, device=d),
    "progressive": lambda d: experiments.run_progressive_protocol(n_seeds=12, device=d),
    "benchmark": lambda d: experiments.run_architecture_benchmark(n_seeds=5, device=d),
    "splits": lambda d: experiments.run_split_sensitivity(n_seeds=6, device=d),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(STEPS), default=list(STEPS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not (paths.DATA_PROCESSED / "dataset_index.csv").exists():
        sys.exit("data/processed/dataset_index.csv not found: run scripts/prepare_data.py first")
    for name in args.only:
        print(f"\n=== {name}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        csv = STEPS[name](args.device)
        print("results:", Path(csv).relative_to(paths.REPO))


if __name__ == "__main__":
    main()
