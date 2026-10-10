"""Baseline category suggester: five scorers fused, evaluated on a temporal split.

Usage:
    python3 baseline.py --data data              # uses candidates.py output if present
    python3 baseline.py --data data --no-candidates

Each image is scored against the categories seen in training plus its own search
candidates from candidates.py (never another image's). Scorers, each 0..1:
    img_knn   similar images (by CLIP image embedding) vote with their categories
    img_txt   image vs category name, CLIP zero-shot
    meta_knn  images with similar title + description vote with their categories
    meta_txt  title + description vs category name
    search    rank of the category in the Commons search for the image's title

Splits by upload time (oldest 70% train, next 10% validation, newest 20% test),
drops test images that are near-duplicates of a train image, picks the fusion
weights on validation, and reports P@k, R@k and macro mAP on test.
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

KS = (1, 5, 10)


def load(data_dir):
    recs = [json.loads(line) for line in open(os.path.join(data_dir, "images.jsonl"), encoding="utf-8")]
    X = np.load(os.path.join(data_dir, "image_emb.npy"))
    M = np.load(os.path.join(data_dir, "meta_emb.npy"))
    cats = json.load(open(os.path.join(data_dir, "categories.json"), encoding="utf-8"))
    T = np.load(os.path.join(data_dir, "text_emb.npy"))
    return recs, X, M, cats, T


def load_candidates(data_dir, min_files):
    """{image title: [category, ...]} and {category: text embedding}, or None."""
    path = os.path.join(data_dir, "candidates.json")
    if not os.path.exists(path):
        return None, {}
    raw = json.load(open(path, encoding="utf-8"))
    # A category whose only file is the image itself was most likely created for
    # that upload, so it would not have existed to be suggested. Require min_files.
    cands = {t: [c for c, files in cs if files >= min_files] for t, cs in raw.items()}
    names = json.load(open(os.path.join(data_dir, "cand_categories.json"), encoding="utf-8"))
    emb = np.load(os.path.join(data_dir, "cand_text_emb.npy"))
    return cands, dict(zip(names, emb))


def temporal_split(recs, X, val_frac, test_frac, dup_thresh):
    order = np.argsort([r["timestamp"] for r in recs])
    n = len(order)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    train = order[: n - n_test - n_val]
    val = order[n - n_test - n_val : n - n_test]
    test = order[n - n_test :]

    # Near-duplicate removal: re-uploads and burst shots leak across a time split.
    def dedup(idx):
        sims = X[idx] @ X[train].T
        keep = sims.max(1) < dup_thresh
        return idx[keep], int((~keep).sum())

    val, _ = dedup(val)
    test, n_dup = dedup(test)
    return train, val, test, n_dup


def label_matrix(recs, idx, vocab_index):
    Y = np.zeros((len(idx), len(vocab_index)), dtype=bool)
    for row, i in enumerate(idx):
        for c in recs[i]["categories"]:
            j = vocab_index.get(c)
            if j is not None:
                Y[row, j] = True
    return Y


class Columns:
    """Score-matrix columns: the shared train vocabulary, then a per-row block of
    that image's own extra candidates (padded to the same width)."""

    def __init__(self, vocab, extra):
        self.vocab, self.extra = vocab, extra
        self.V = len(vocab)
        self.P = max((len(e) for e in extra), default=0)

    def name(self, r, j):
        if j < self.V:
            return self.vocab[j]
        e = self.extra[r]
        return e[j - self.V] if j - self.V < len(e) else None

    def col(self, r, c, vocab_index):
        if c in vocab_index:
            return vocab_index[c]
        return self.V + self.extra[r].index(c) if c in self.extra[r] else None


def knn_scores(Q, E, Ytrain, k, tau, P):
    """s_knn(c|x) = sum over neighbours of exp(sim/tau) * [c in Y_i], normalized."""
    sims = Q @ E.T
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    w = np.exp(np.take_along_axis(sims, nn, 1) / tau)
    w /= w.sum(1, keepdims=True)
    s = np.einsum("qk,qkc->qc", w, Ytrain[nn].astype(np.float32))
    return np.hstack([s, np.zeros((len(Q), P), np.float32)])


def text_scores(Q, Tvocab, Textra, pad, temp):
    """Softmax over each row's category-name similarities (vocab + own candidates)."""
    logits = np.hstack([Q @ Tvocab.T, np.einsum("nd,npd->np", Q, Textra)]) / temp
    logits[:, Tvocab.shape[0]:][pad] = -np.inf
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def precision_recall_at_k(S, true_sets, cols):
    """Sample-averaged P@k and R@k. true_sets keeps labels the model cannot suggest,
    so they still count against recall."""
    top = np.argpartition(-S, max(KS), axis=1)[:, : max(KS)]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)
    out = {}
    for k in KS:
        p, r = [], []
        for row, (js, truth) in enumerate(zip(top[:, :k], true_sets)):
            hits = sum(cols.name(row, j) in truth for j in js)
            p.append(hits / k)
            r.append(hits / len(truth))
        out[f"P@{k}"], out[f"R@{k}"] = np.mean(p), np.mean(r)
    return out


def macro_map(S, true_sets, cols, vocab_index):
    """Mean over categories of average precision, for every true category that is
    suggestible for at least one test image (rows where it is not get -inf)."""
    rows_of = defaultdict(list)
    for r, truth in enumerate(true_sets):
        for c in truth:
            rows_of[c].append(r)
    available = set(cols.vocab) | {c for e in cols.extra for c in e}
    aps = []
    for c in rows_of:
        if c not in available:
            continue
        scores = np.full(len(true_sets), -np.inf)
        for r in range(len(true_sets)):
            j = cols.col(r, c, vocab_index)
            if j is not None:
                scores[r] = S[r, j]
        y = np.zeros(len(true_sets), bool)
        y[rows_of[c]] = True
        y = y[np.argsort(-scores, kind="stable")]
        aps.append((np.cumsum(y)[y] / (np.where(y)[0] + 1)).mean())
    return float(np.mean(aps))


def simplex(n, steps):
    """All weight vectors of length n with entries in multiples of 1/steps summing to 1."""
    for c in itertools.product(range(steps + 1), repeat=n - 1):
        if sum(c) <= steps:
            yield np.array([*c, steps - sum(c)]) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--k", type=int, default=20, help="neighbours for k-NN voting")
    ap.add_argument("--tau", type=float, default=0.05, help="k-NN similarity temperature")
    ap.add_argument("--temp", type=float, default=0.01, help="zero-shot softmax temperature (CLIP uses 0.01)")
    ap.add_argument("--dup-thresh", type=float, default=0.95, help="cosine above this = near-duplicate")
    ap.add_argument("--no-candidates", action="store_true", help="ignore candidates.py output")
    ap.add_argument("--min-cand-files", type=int, default=2,
                    help="drop search candidates with fewer files (likely created for this upload)")
    ap.add_argument("--objective", default="P@1+R@10", choices=["R@5", "P@1", "R@10", "P@1+R@10"],
                    help="what the fusion weights maximize on validation")
    ap.add_argument("--examples", type=int, default=4, help="test images to print suggestions for")
    args = ap.parse_args()

    recs, X, M, cats, T = load(args.data)
    train, val, test, n_dup = temporal_split(recs, X, 0.1, 0.2, args.dup_thresh)
    print(f"{len(recs)} images: train {len(train)}, val {len(val)}, test {len(test)} "
          f"({n_dup} near-duplicates of train removed from test)")
    print(f"test uploads from {recs[test[0]]['timestamp'][:10]} onward")

    cands, cand_emb = (None, {}) if args.no_candidates else load_candidates(args.data, args.min_cand_files)
    scorers = ["img_knn", "img_txt", "meta_knn", "meta_txt"] + (["search"] if cands else [])

    vocab = sorted({c for i in train for c in recs[i]["categories"]})
    vocab_index = {c: j for j, c in enumerate(vocab)}
    cat_index = {c: j for j, c in enumerate(cats)}
    Tvocab = T[[cat_index[c] for c in vocab]]
    Ytrain = label_matrix(recs, train, vocab_index)
    print(f"vocabulary: {len(vocab)} categories seen in train"
          + (f" + each image's own search candidates (>= {args.min_cand_files} files)" if cands else "") + "\n")

    def all_scores(idx):
        found = [cands.get(recs[i]["title"], []) if cands else [] for i in idx]
        cols = Columns(vocab, [[c for c in f if c not in vocab_index] for f in found])
        D = T.shape[1]
        Textra = np.zeros((len(idx), cols.P, D), np.float32)
        pad = np.ones((len(idx), cols.P), bool)
        for r, e in enumerate(cols.extra):
            for p, c in enumerate(e):
                Textra[r, p], pad[r, p] = cand_emb[c], False
        s = {
            "img_knn": knn_scores(X[idx], X[train], Ytrain, args.k, args.tau, cols.P),
            "img_txt": text_scores(X[idx], Tvocab, Textra, pad, args.temp),
            "meta_knn": knn_scores(M[idx], M[train], Ytrain, args.k, args.tau, cols.P),
            "meta_txt": text_scores(M[idx], Tvocab, Textra, pad, args.temp),
        }
        if cands:
            srch = np.zeros((len(idx), cols.V + cols.P), np.float32)
            for r, f in enumerate(found):
                for rank, c in enumerate(f):
                    srch[r, cols.col(r, c, vocab_index)] = 1 / (rank + 1)
            s["search"] = srch / np.maximum(srch.sum(1, keepdims=True), 1e-9)
        return s, cols

    def objective(m):
        return m["P@1"] + m["R@10"] if args.objective == "P@1+R@10" else m[args.objective]

    def fuse(scores, w):
        return sum(wi * scores[name] for wi, name in zip(w, scorers))

    # Pick fusion weights on validation (grid over the simplex, step 0.1), never on test.
    val_scores, val_cols = all_scores(val)
    val_truth = [set(recs[i]["categories"]) for i in val]
    best_w = max(simplex(len(scorers), 10),
                 key=lambda w: objective(precision_recall_at_k(fuse(val_scores, w), val_truth, val_cols)))
    print(f"fusion weights picked on validation (maximizing {args.objective}): "
          + ", ".join(f"{n}={w:.1f}" for n, w in zip(scorers, best_w)) + "\n")

    test_scores, cols = all_scores(test)
    test_truth = [set(recs[i]["categories"]) for i in test]
    image_only = max(simplex(2, 10), key=lambda w: objective(precision_recall_at_k(
        fuse(val_scores, [*w] + [0] * (len(scorers) - 2)), val_truth, val_cols)))
    results = {name: test_scores[name] for name in scorers}
    results["image only (fused)"] = fuse(test_scores, [*image_only] + [0] * (len(scorers) - 2))
    results["all (fused)"] = fuse(test_scores, best_w)

    # Every image carries its seed, so also check the labels that are not the seed.
    def non_seed_recall(S):
        top10 = np.argpartition(-S, 10, axis=1)[:, :10]
        rs = []
        for r, js in enumerate(top10):
            truth = test_truth[r] - {recs[test[r]].get("seed")}
            if truth:
                rs.append(len({cols.name(r, j) for j in js} & truth) / len(truth))
        return np.mean(rs)

    names = ["P@1", "P@5", "R@5", "R@10"]
    print(f"{'method':<20}" + "".join(f"{c:>8}" for c in names + ["mAP"]) + f"{'R@10 non-seed':>15}")
    for name, S in results.items():
        m = precision_recall_at_k(S, test_truth, cols)
        print(f"{name:<20}" + "".join(f"{m[x]:8.3f}" for x in names)
              + f"{macro_map(S, test_truth, cols, vocab_index):8.3f}{non_seed_recall(S):15.3f}")

    S = results["all (fused)"]
    print("\nall (fused), by seed type:")
    groups = defaultdict(list)
    for row, i in enumerate(test):
        groups[recs[i].get("seed_type") or "none"].append(row)
    for kind, rows in sorted(groups.items()):
        sub = Columns(vocab, [cols.extra[r] for r in rows])
        sub.P = cols.P
        before = precision_recall_at_k(results["image only (fused)"][rows], [test_truth[r] for r in rows], sub)
        m = precision_recall_at_k(S[rows], [test_truth[r] for r in rows], sub)
        print(f"  {kind:<10} n={len(rows):<5} P@1 {m['P@1']:.3f}  R@10 {m['R@10']:.3f}"
              f"   (image only: P@1 {before['P@1']:.3f}  R@10 {before['R@10']:.3f})")

    for e in range(args.examples):
        r = e * len(test) // max(args.examples, 1)
        rec, row = recs[test[r]], S[r]
        print(f"\n{rec['title']}")
        print(f"  true: {', '.join(c.removeprefix('Category:') for c in rec['categories'])}")
        for j in np.argsort(-row)[:5]:
            c = cols.name(r, j)
            mark = "+" if c in rec["categories"] else " "
            print(f"  {mark} {row[j]:.2f}  {(c or '-').removeprefix('Category:')}")


if __name__ == "__main__":
    main()
