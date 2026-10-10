"""Download thumbnails from a fetch_commons.py JSONL file and embed them with CLIP.

Usage:
    pip install torch open_clip_torch pillow numpy
    python3 embed.py --data commons_data.jsonl --out data

Writes into --out:
    thumbs/            cached thumbnails (re-runs skip what is already downloaded)
    images.jsonl       the records that downloaded OK, row i <-> image_emb.npy[i]
    image_emb.npy      (N, D) float32, L2-normalized image embeddings
    categories.json    every category seen in the data, row j <-> text_emb.npy[j]
    text_emb.npy       (C, D) float32, L2-normalized text embeddings of category names
    meta_emb.npy       (N, D) float32, L2-normalized text embeddings of each image's
                       cleaned title + description, row i <-> images.jsonl[i]
"""
import argparse
import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open_clip
import torch
from PIL import Image

from fetch_commons import USER_AGENT

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"  # open weights, trained on LAION-2B


def thumb_path(thumb_dir, title):
    return os.path.join(thumb_dir, hashlib.md5(title.encode()).hexdigest() + ".jpg")


def download(url, path):
    """Download url to path; return True on success. Retries on 429/5xx."""
    if os.path.exists(path):
        return True
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return True
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                return False
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    return False


def category_prompt(cat):
    # "Category:Bridges in Paris" -> "a photo of bridges in paris"
    return "a photo of " + cat.removeprefix("Category:").lower()


def meta_text(rec):
    """Title + description as one string, minus file-name noise.

    "File:Dog_at_Lifou (12345).jpg" -> "Dog at Lifou"
    """
    title = rec["title"].removeprefix("File:")
    title = re.sub(r"\.[A-Za-z0-9]{3,4}$", "", title)        # extension
    title = re.sub(r"\(\d+\)|\bIMG[_ -]?\d+|\bDSC\w*\d+", " ", title)  # flickr ids, camera names
    desc = html.unescape(rec.get("description") or "")
    desc = desc.removeprefix("500px provided description:")
    text = re.sub(r"[_\s]+", " ", f"{title}. {desc}").strip(" .")
    return text or "photo"


def embed_texts(model, tokenizer, texts, device, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        tokens = tokenizer(texts[i:i + batch]).to(device)  # truncates to CLIP's 77 tokens
        with torch.no_grad():
            z = model.encode_text(tokens)
        out.append(torch.nn.functional.normalize(z, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="commons_data.jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads; keep small to be polite")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    thumb_dir = os.path.join(args.out, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    records = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    records = [r for r in records if r.get("thumb_url")]
    paths = [thumb_path(thumb_dir, r["title"]) for r in records]

    print(f"Downloading {len(records)} thumbnails...")
    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(pool.map(download, [r["thumb_url"] for r in records], paths))
    print(f"  {sum(ok)} ok, {len(ok) - sum(ok)} failed")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model.eval()

    kept, embs = [], []
    todo = [(r, p) for r, p, good in zip(records, paths, ok) if good]
    print(f"Embedding {len(todo)} images on {device}...")
    for i in range(0, len(todo), args.batch):
        batch, tensors = [], []
        for r, p in todo[i:i + args.batch]:
            try:
                tensors.append(preprocess(Image.open(p).convert("RGB")))
                batch.append(r)
            except Exception as e:  # corrupt or unsupported file
                print(f"  skip {r['title']}: {e}")
        if not tensors:
            continue
        with torch.no_grad():
            z = model.encode_image(torch.stack(tensors).to(device))
        embs.append(torch.nn.functional.normalize(z, dim=-1).float().cpu().numpy())
        kept += batch
        print(f"  {len(kept)}/{len(todo)}", end="\r")
    print()

    categories = sorted({c for r in kept for c in r["categories"]})
    print(f"Embedding {len(categories)} category names...")
    text_emb = embed_texts(model, tokenizer, [category_prompt(c) for c in categories], device)
    print(f"Embedding {len(kept)} titles + descriptions...")
    meta_emb = embed_texts(model, tokenizer, [meta_text(r) for r in kept], device)

    np.save(os.path.join(args.out, "image_emb.npy"), np.concatenate(embs))
    np.save(os.path.join(args.out, "text_emb.npy"), text_emb)
    np.save(os.path.join(args.out, "meta_emb.npy"), meta_emb)
    with open(os.path.join(args.out, "images.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False)
    print(f"Done: {len(kept)} images, {len(categories)} categories -> {args.out}/")


if __name__ == "__main__":
    main()
