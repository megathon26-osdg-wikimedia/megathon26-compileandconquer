"""Latency, throughput and memory of the suggestion pipeline.

Times each stage of suggesting categories for one upload, on a sample of test
images: CLIP image + text encoding (GPU and CPU), scoring and fusion, and the
live Commons search call. Then measures batched throughput and index memory.

Usage:
    python3 benchmark.py --data data
"""
import os
import sys
import time

import numpy as np
import open_clip
import torch
from PIL import Image

from baseline import parse_args, prepare
from candidates import query_words, search_categories
from embed import MODEL, PRETRAINED, meta_text, thumb_path

SAMPLE = 100       # images timed one at a time
SEARCH_CALLS = 10  # live API calls (kept small to be polite)


def ms(times):
    t = np.array(times) * 1000
    return f"median {np.median(t):7.1f} ms   p95 {np.percentile(t, 95):7.1f} ms"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_encoders(device, recs, idx, thumb_dir):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL)
    img_t, txt_t = [], []
    with torch.no_grad():
        for n, i in enumerate(idx):
            t0 = time.perf_counter()
            x = preprocess(Image.open(thumb_path(thumb_dir, recs[i]["title"])).convert("RGB"))[None].to(device)
            model.encode_image(x)
            sync(device)
            t1 = time.perf_counter()
            model.encode_text(tokenizer([meta_text(recs[i])]).to(device))
            sync(device)
            t2 = time.perf_counter()
            if n >= 5:  # first calls include warm-up
                img_t.append(t1 - t0)
                txt_t.append(t2 - t1)
        # Batched throughput: what a queue of uploads would see.
        batch = torch.stack([preprocess(Image.open(thumb_path(thumb_dir, recs[i]["title"])).convert("RGB"))
                             for i in idx[:64]]).to(device)
        model.encode_image(batch)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(3):
            model.encode_image(batch)
        sync(device)
        per_sec = 3 * len(batch) / (time.perf_counter() - t0)
    params = sum(p.numel() for p in model.parameters())
    return img_t, txt_t, per_sec, params


def main():
    args = parse_args(sys.argv[1:] + ["--examples", "0"])
    print("Preparing...")
    ctx = prepare(args, verbose=False)
    rng = np.random.default_rng(0)
    idx = list(rng.choice(ctx.test, SAMPLE, replace=False))
    thumb_dir = os.path.join(args.data, "thumbs")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else
                         ["mps"] if torch.backends.mps.is_available() else [])
    enc = {d: time_encoders(d, ctx.recs, idx, thumb_dir) for d in devices}

    score_t = []
    for i in idx:
        t0 = time.perf_counter()
        s, _ = ctx.all_scores([i])
        S = ctx.fuse(s, ctx.best_w)
        np.argpartition(-S[0], 10)[:10]
        score_t.append(time.perf_counter() - t0)

    search_t = []
    for i in idx[:SEARCH_CALLS]:
        words = query_words(ctx.recs[i])
        if words:
            t0 = time.perf_counter()
            search_categories(words, 30)
            search_t.append(time.perf_counter() - t0)

    print(f"\nPer-upload latency ({SAMPLE} test images, one at a time)")
    for d, (img_t, txt_t, _, _) in enc.items():
        print(f"  CLIP image encode ({d:<4})        {ms(img_t)}")
        print(f"  CLIP text encode  ({d:<4})        {ms(txt_t)}")
    print(f"  scoring + fusion (numpy, CPU)    {ms(score_t)}")
    print(f"  Commons search API (network)     {ms(search_t)}   ({len(search_t)} live calls)")
    fast = devices[-1]
    total = (np.median(enc[fast][0]) + np.median(enc[fast][1]) + np.median(score_t) + np.median(search_t)) * 1000
    print(f"  end to end, median ({fast})       ~{total:.0f} ms")

    print("\nThroughput (batch of 64 images)")
    for d, (_, _, per_sec, params) in enc.items():
        print(f"  {d:<4}  {per_sec:7.1f} images/s  = {per_sec * 86400 / 1e6:5.1f} M images/day")
    print(f"  model: {MODEL} ({PRETRAINED}), {enc[fast][3] / 1e6:.0f} M parameters")

    n, d = ctx.X[ctx.train].shape
    flat = 2 * d * 4  # image + title vectors, float32
    print("\nIndex memory")
    print(f"  now: {n} training images x 2 vectors x {d} dims, float32 = {n * flat / 1e6:.1f} MB")
    print(f"  per image: {flat} B float32, {flat // 2} B float16, 128 B with 64-byte product quantization")
    print(f"  at 100 M images: {100e6 * flat / 1e9:.0f} GB float32, {100e6 * flat / 2e9:.0f} GB float16, "
          f"{100e6 * 128 / 1e9:.1f} GB with PQ")
    print("  (scoring here is brute force over the training set; at that scale an ANN index such as "
          "FAISS IVF-PQ would replace it)")


if __name__ == "__main__":
    main()
