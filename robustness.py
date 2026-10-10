"""How much accuracy is lost on low-quality images?

Degrades each cached test thumbnail (blur, heavy JPEG, low resolution), re-embeds
it with CLIP and scores it against the clean training index, with the fusion
weights picked on clean validation data. Titles and descriptions are untouched,
since a bad photo still has its text.

Usage:
    python3 robustness.py --data data
"""
import io
import os
import sys

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFilter

from baseline import parse_args, prepare, precision_recall_at_k
from embed import MODEL, PRETRAINED, thumb_path

DEGRADATIONS = {
    "clean": lambda im: im,
    "blur (radius 3)": lambda im: im.filter(ImageFilter.GaussianBlur(3)),
    "JPEG quality 10": lambda im: jpeg(im, 10),
    "low-res 96 px": lambda im: downscale(im, 96),
    "low-res 48 px": lambda im: downscale(im, 48),
}


def jpeg(im, quality):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def downscale(im, short_side):
    """Shrink so the short side is short_side px; CLIP's preprocessing scales it back up."""
    scale = short_side / min(im.size)
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)


def main():
    args = parse_args(sys.argv[1:] + ["--examples", "0"])
    print("Preparing (split, vocabulary, fusion weights on clean validation)...")
    ctx = prepare(args, verbose=False)
    test = ctx.test

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    model.eval()
    thumbs = [Image.open(thumb_path(os.path.join(args.data, "thumbs"), ctx.recs[i]["title"])).convert("RGB")
              for i in test]

    rows = []
    for name, degrade in DEGRADATIONS.items():
        if name == "clean":
            Xq = ctx.X[test]  # the embeddings baseline.py uses
        else:
            embs = []
            for b in range(0, len(thumbs), 64):
                batch = torch.stack([preprocess(degrade(im)) for im in thumbs[b:b + 64]]).to(device)
                with torch.no_grad():
                    z = model.encode_image(batch)
                embs.append(torch.nn.functional.normalize(z, dim=-1).float().cpu().numpy())
            Xq = np.concatenate(embs)
        scores, cols = ctx.all_scores(test, Xq)
        img = precision_recall_at_k(ctx.fuse(scores, ctx.image_only_w), ctx.test_truth, cols)
        full = precision_recall_at_k(ctx.fuse(scores, ctx.best_w), ctx.test_truth, cols)
        rows.append((name, img, full))
        print(f"  done: {name}")

    base_img, base_full = rows[0][1], rows[0][2]
    print(f"\n{len(test)} test images; fusion weights from clean validation data\n")
    print(f"{'condition':<18}{'image only':>22}{'all five scorers':>26}")
    print(f"{'':<18}{'P@1':>11}{'R@10':>11}{'P@1':>13}{'R@10':>13}")
    for name, img, full in rows:
        print(f"{name:<18}{img['P@1']:>11.3f}{img['R@10']:>11.3f}{full['P@1']:>13.3f}{full['R@10']:>13.3f}"
              + ("" if name == "clean" else
                 f"   (P@1 change: image only {img['P@1'] - base_img['P@1']:+.3f},"
                 f" all five {full['P@1'] - base_full['P@1']:+.3f})"))


if __name__ == "__main__":
    main()
