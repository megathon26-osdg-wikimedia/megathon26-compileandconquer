"""Find candidate categories for each val/test image by searching Commons with its title.

The k-NN scorers can only suggest categories seen in training, but most real labels
are specific places, events and people that may never have appeared there. This
script asks the Commons search API for categories matching the words in each
image's title (falling back to its description), so baseline.py can rank them.

Usage:
    python3 candidates.py --data data

Writes into --data:
    search_cache.json   query -> results, so re-runs and interruptions cost nothing
    candidates.json     image title -> [[category, files], ...] in search-rank order
    cand_categories.json, cand_text_emb.npy   CLIP text embeddings of all candidate names
"""
import argparse
import json
import os
import re

import numpy as np
import open_clip
import torch

from baseline import temporal_split
from embed import MODEL, PRETRAINED, category_prompt, embed_texts
from fetch_commons import api_get

# Words that match thousands of categories and say nothing about the image.
STOP = set("""
a an the of in at on by and or for to with from into near over under is are was this that
de del la las el los en le les du des et un une der die das und im am van het een op di il della
da do dos das e o y photo photos image picture view img dsc jpg jpeg png untitled file copy
provided description own work
""".split())


def query_words(rec, max_words=8):
    title = re.sub(r"\.[A-Za-z0-9]{3,4}$", "", rec["title"].removeprefix("File:"))
    words = [w for w in re.findall(r"[^\W\d_]{3,}", title) if w.lower() not in STOP]
    if len(words) < 2:  # junk titles like "25P0679 13": use the description instead
        desc = re.findall(r"[^\W\d_]{3,}", rec.get("description") or "")
        words += [w for w in desc if w.lower() not in STOP]
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out[:max_words]


def search_categories(words, limit):
    """Category pages matching any of the words, best first, hidden ones dropped."""
    data = api_get({
        "action": "query",
        "generator": "search",
        "gsrnamespace": "14",
        "gsrsearch": " OR ".join(words),
        "gsrlimit": str(limit),
        "prop": "categoryinfo",
    })
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p["index"])
    return [[p["title"], p.get("categoryinfo", {}).get("files", 0)]
            for p in pages if not p.get("categoryinfo", {}).get("hidden")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=30, help="candidates per image")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(os.path.join(args.data, "images.jsonl"), encoding="utf-8")]
    X = np.load(os.path.join(args.data, "image_emb.npy"))
    _, val, test, _ = temporal_split(recs, X, 0.1, 0.2, 0.95)

    cache_path = os.path.join(args.data, "search_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    candidates, failed = {}, 0
    todo = [recs[i] for i in [*val, *test]]
    for n, rec in enumerate(todo, 1):
        words = query_words(rec)
        key = " OR ".join(words)
        if words and key not in cache:
            try:
                cache[key] = search_categories(words, args.limit)
            except RuntimeError as e:  # still failing after retries: skip, a re-run fills it in
                print(f"  search failed for {rec['title']}: {e}")
                failed += 1
                continue
            if len(cache) % 50 == 0:
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
        candidates[rec["title"]] = cache.get(key, [])
        print(f"  {n}/{len(todo)}", end="\r", flush=True)
    print()
    if failed:
        print(f"{failed} searches failed; run again to fill them in (results are cached)")
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(candidates, open(os.path.join(args.data, "candidates.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    names = sorted({c for cs in candidates.values() for c, _ in cs})
    print(f"Embedding {len(names)} candidate category names...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    model.eval()
    emb = embed_texts(model, open_clip.get_tokenizer(MODEL), [category_prompt(c) for c in names], device)
    np.save(os.path.join(args.data, "cand_text_emb.npy"), emb)
    json.dump(names, open(os.path.join(args.data, "cand_categories.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    hit = [len({c for c, _ in candidates[r["title"]]} & (set(r["categories"]) - {r.get("seed")}))
           / max(len(set(r["categories"]) - {r.get("seed")}), 1)
           for r in todo if set(r["categories"]) - {r.get("seed")}]
    print(f"Done. Non-seed labels found among candidates: {np.mean(hit):.1%} (upper bound for ranking)")


if __name__ == "__main__":
    main()
