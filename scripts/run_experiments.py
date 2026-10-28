"""
Unified CLI to run baseline suite, adaptive evaluations, and ablations.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner CLI")
    parser.add_argument("--mode", choices=["all", "baselines", "ablations", "figures"], default="all")
    args = parser.parse_args()

    if args.mode in ["all", "baselines"]:
        from scripts.evaluate_all_methods import main as eval_main
        eval_main()

    if args.mode in ["all", "ablations"]:
        from scripts.run_ablations import main as abl_main
        abl_main()

    if args.mode in ["all", "figures"]:
        from scripts.make_figures import main as fig_main
        fig_main()


if __name__ == "__main__":
    main()
