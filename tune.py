"""Tune k-NN size and temperatures on validation data only.

For each setting, the stacked fusion is fit on one half of the validation split and
scored on the other half; test is never touched. Prints every setting, best first.

Usage:
    python3 tune.py --data data
"""
import itertools
import sys
import time

from baseline import parse_args, precision_recall_at_k, prepare

GRID = {
    "k": [10, 20, 40],
    "tau": [0.02, 0.05, 0.1],
    "temp": [0.01, 0.03],
}


def main():
    base = parse_args(sys.argv[1:] + ["--skip-grid", "--examples", "0"])
    results = []
    combos = list(itertools.product(*GRID.values()))
    for n, values in enumerate(combos, 1):
        args = parse_args(sys.argv[1:] + ["--skip-grid", "--examples", "0"])
        for key, v in zip(GRID, values):
            setattr(args, key, v)
        t0 = time.time()
        ctx = prepare(args, verbose=False, holdout=True)
        scores, cols = ctx.all_scores(ctx.holdout)
        truth = [set(ctx.recs[i]["categories"]) for i in ctx.holdout]
        m = precision_recall_at_k(ctx.stacked(scores, cols), truth, cols)
        results.append((m["P@1"] + m["R@10"], m["P@1"], m["R@10"], dict(zip(GRID, values))))
        print(f"[{n}/{len(combos)}] {dict(zip(GRID, values))}  P@1 {m['P@1']:.3f}  R@10 {m['R@10']:.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nBest first (held-out half of validation, {len(ctx.holdout)} images):")
    for score, p1, r10, setting in sorted(results, key=lambda x: -x[0])[:10]:
        print(f"  P@1 {p1:.3f}  R@10 {r10:.3f}  " + "  ".join(f"--{k} {v}" for k, v in setting.items()))
    defaults = {k: getattr(base, k) for k in GRID}
    d = next(x for x in results if x[3] == defaults)
    print(f"\ndefaults ({'  '.join(f'--{k} {v}' for k, v in defaults.items())}): P@1 {d[1]:.3f}  R@10 {d[2]:.3f}")


if __name__ == "__main__":
    main()
