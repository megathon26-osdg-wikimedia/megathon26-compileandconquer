"""Local demo: suggest Commons categories for an image, explain them, take feedback.

Usage:
    python3 demo.py            # then open http://localhost:8000

Query by a random test image, a Commons file name or link, or an uploaded photo
with a title. Each suggestion shows which scorers back it and the similar images
behind it. Accepting adds the image to the in-memory index at once (no
retraining); every accept/reject is appended to data/feedback.jsonl.

Uses everything in --data built by embed.py and candidates.py, plus
stack_model.json (the learned blend) written by baseline.py; falls back to the
fixed weights in fusion_weights.json if the stacked model is missing.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import socket
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import open_clip
import torch
from PIL import Image

from baseline import is_process, temporal_split
from candidates import query_words, search_categories
from embed import MODEL, PRETRAINED, category_prompt, download, meta_text, thumb_path
from fetch_commons import THUMB_WIDTH, api_get, strip_html

K, TAU, TEMP, MIN_CAND_FILES, TOP = 20, 0.1, 0.03, 2, 10  # as tuned by tune.py
HERE = os.path.dirname(os.path.abspath(__file__))


class Suggester:
    def __init__(self, data):
        self.data = data
        self.lock = threading.Lock()
        self.recs = [json.loads(l) for l in open(os.path.join(data, "images.jsonl"), encoding="utf-8")]
        self.X = np.load(os.path.join(data, "image_emb.npy"))
        self.M = np.load(os.path.join(data, "meta_emb.npy"))
        self.cats = json.load(open(os.path.join(data, "categories.json"), encoding="utf-8"))
        self.T = np.load(os.path.join(data, "text_emb.npy"))
        self.cat_index = {c: j for j, c in enumerate(self.cats)}
        self.Y = np.zeros((len(self.recs), len(self.cats)), bool)
        for i, r in enumerate(self.recs):
            for c in r["categories"]:
                self.Y[i, self.cat_index[c]] = True
        self.counts = self.Y.sum(0)  # images per category in the index
        self.process = np.array([is_process(c) for c in self.cats])  # workflow categories, never suggested
        names = json.load(open(os.path.join(data, "cand_categories.json"), encoding="utf-8"))
        self.name_emb = dict(zip(names, np.load(os.path.join(data, "cand_text_emb.npy"))))
        self.weights = json.load(open(os.path.join(data, "fusion_weights.json")))
        stack_path = os.path.join(data, "stack_model.json")
        self.stack = json.load(open(stack_path)) if os.path.exists(stack_path) else None
        if self.stack:
            self.stack.update({k: np.array(self.stack[k]) for k in ("mu", "sd", "w")})
        _, _, self.test, _ = temporal_split(self.recs, self.X, 0.1, 0.2, 0.95)

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=PRETRAINED, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL)
        self.sessions = {}  # query id -> what is needed to add it to the index on accept
        self.feedback_path = os.path.join(data, "feedback.jsonl")

    # ---- encoding -------------------------------------------------------
    def encode_image(self, img):
        with torch.no_grad():
            z = self.model.encode_image(self.preprocess(img.convert("RGB"))[None].to(self.device))
        return torch.nn.functional.normalize(z, dim=-1)[0].float().cpu().numpy()

    def encode_texts(self, texts):
        with torch.no_grad():
            z = self.model.encode_text(self.tokenizer(texts).to(self.device))
        return torch.nn.functional.normalize(z, dim=-1).float().cpu().numpy()

    # ---- the five scorers, for one query ---------------------------------
    def knn(self, q, E, exclude):
        sims = E @ q
        if exclude is not None:
            sims[exclude] = -np.inf
        nn = np.argpartition(-sims, K)[:K]
        nn = nn[np.argsort(-sims[nn])]
        w = np.exp(sims[nn] / TAU)
        w /= w.sum()
        return w @ self.Y[nn].astype(np.float32), nn, sims[nn]

    @staticmethod
    def softmax_scores(q, Tall, allowed):
        logits = Tall[allowed] @ q / TEMP
        p = np.zeros(len(Tall), np.float32)
        p[allowed] = np.exp(logits - logits.max())
        return p / p.sum()

    def suggest(self, img, rec, exclude=None):
        """rec = {title, description, ...}; exclude = row of this image in the index, if any."""
        t0 = time.perf_counter()
        x = self.encode_image(img)
        m = self.encode_texts([meta_text(rec)])[0]

        words = query_words(rec)
        found = []
        if words:
            try:
                found = [c for c, files in search_categories(words, 30)
                         if files >= MIN_CAND_FILES and not is_process(c)]
            except Exception as e:  # search is optional; the other four scorers still work
                print(f"search failed: {e}")
        extra = [c for c in found if c not in self.cat_index]
        missing = [c for c in extra if c not in self.name_emb]
        if missing:
            for c, e in zip(missing, self.encode_texts([category_prompt(c) for c in missing])):
                self.name_emb[c] = e
        names = self.cats + extra
        Tall = np.vstack([self.T] + [self.name_emb[c][None] for c in extra]) if extra else self.T
        pad = np.zeros(len(extra), np.float32)

        img_knn, img_nn, img_sims = self.knn(x, self.X, exclude)
        meta_knn, meta_nn, meta_sims = self.knn(m, self.M, exclude)
        search = np.zeros(len(names), np.float32)
        for rank, c in enumerate(found):
            col = self.cat_index[c] if c in self.cat_index else len(self.cats) + extra.index(c)
            search[col] = 1 / (rank + 1)
        search /= max(search.sum(), 1e-9)
        # As in baseline.py: a category is suggestible if some indexed image uses it or
        # search found it for this image, and it is not a workflow category.
        counts = np.concatenate([self.counts, pad])
        allowed = ((counts > 0) | (search > 0)) & ~np.concatenate([self.process, pad.astype(bool)])
        parts = {
            "img_knn": np.concatenate([img_knn, pad]),
            "img_txt": self.softmax_scores(x, Tall, allowed),
            "meta_knn": np.concatenate([meta_knn, pad]),
            "meta_txt": self.softmax_scores(m, Tall, allowed),
            "search": search,
        }
        if self.stack:
            S, contribs = self.stacked(parts, allowed, counts)
        else:
            S = sum(self.weights[k] * v for k, v in parts.items())
            contribs = {k: self.weights[k] * v for k, v in parts.items()}
        S[~allowed] = -1
        top = np.argsort(-S)[:TOP]

        out = []
        for j in top:
            c = names[j]
            contrib = {k: float(contribs[k][j]) for k in parts}
            backers = []
            if j < len(self.cats):
                for nn, sims in ((img_nn, img_sims), (meta_nn, meta_sims)):
                    for i, s in zip(nn, sims):
                        if self.Y[i, j] and i not in [b["row"] for b in backers]:
                            backers.append({"row": int(i), "sim": float(s)})
            out.append({
                "category": c,
                "score": float(S[j]),
                "contrib": contrib,
                "similar": [{"title": self.recs[b["row"]]["title"], "sim": round(b["sim"], 3),
                             "thumb": self.thumb_url(self.recs[b["row"]]["title"])}
                            for b in backers[:3]],
                "in_index": bool(j < len(self.cats)),
            })

        qid = uuid.uuid4().hex[:12]
        self.sessions[qid] = {"x": x, "m": m, "rec": rec, "row": exclude}
        return {
            "id": qid,
            "suggestions": out,
            "search_words": words,
            "ms": round((time.perf_counter() - t0) * 1000),
        }

    def stacked(self, parts, allowed, counts):
        """Learned blend from baseline.py: logistic regression over the union of each
        scorer's top candidates, with the same 17 features. Also returns, per scorer,
        how much it raised the log-odds compared with that scorer saying nothing."""
        st, n = self.stack, len(parts["img_knn"])
        valid = np.flatnonzero(allowed)  # ranks are among suggestible categories only
        ranks, cand = {}, set()
        for k in st["scorers"]:
            order = valid[np.argsort(-parts[k][valid])]
            ranks[k] = np.full(n, n)
            ranks[k][order] = np.arange(len(order))
            cand.update(order[: st["stack_top"]].tolist())
        js = np.array(sorted(cand))
        F, silent = [], []
        for k in st["scorers"]:
            sc = parts[k][js]
            F += [sc, np.log(sc + 1e-4), 1 / (1 + ranks[k][js])]
            silent += [0.0, np.log(1e-4), 0.0]
        F = np.stack(F + [np.log1p(counts[js]), (counts[js] == 0).astype(float)], 1)  # 2nd: only search found it
        w, bias = st["w"][:-1], st["w"][-1]
        S = np.full(n, -1.0)
        S[js] = 1 / (1 + np.exp(-(((F - st["mu"]) / st["sd"]) @ w + bias)))
        m = len(silent)
        lift = (F[:, :m] - silent) / st["sd"][:m] * w[:m]
        contribs = {}
        for i, k in enumerate(st["scorers"]):
            contribs[k] = np.zeros(n)
            contribs[k][js] = np.maximum(lift[:, 3 * i: 3 * i + 3].sum(1), 0)
        return S, contribs

    # ---- feedback: accepted labels join the index at once ----------------
    def feedback(self, qid, category, decision):
        s = self.sessions.get(qid)
        if s is None:
            return {"ok": False, "error": "unknown query; suggest again"}
        with open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "title": s["rec"].get("title"),
                                "category": category, "decision": decision}, ensure_ascii=False) + "\n")
        if decision != "accept":
            return {"ok": True, "note": "logged as a hard negative for the next retrain"}
        if category not in self.cat_index:  # a category found by search: add it to the vocabulary
            self.cat_index[category] = len(self.cats)
            self.cats.append(category)
            self.T = np.vstack([self.T, self.name_emb[category][None]])
            self.Y = np.hstack([self.Y, np.zeros((len(self.Y), 1), bool)])
            self.counts = np.append(self.counts, 0)
            self.process = np.append(self.process, False)
        if s["row"] is None:  # a new image: append it to the index
            self.X = np.vstack([self.X, s["x"][None]])
            self.M = np.vstack([self.M, s["m"][None]])
            self.Y = np.vstack([self.Y, np.zeros((1, self.Y.shape[1]), bool)])
            self.recs.append({**s["rec"], "categories": []})
            s["row"] = len(self.recs) - 1
        if not self.Y[s["row"], self.cat_index[category]]:
            self.counts[self.cat_index[category]] += 1
        self.Y[s["row"], self.cat_index[category]] = True
        self.recs[s["row"]]["categories"] = sorted(set(self.recs[s["row"]]["categories"]) | {category})
        return {"ok": True, "note": f"added to the index ({len(self.recs)} images); no retraining"}

    # ---- queries ---------------------------------------------------------
    def thumb_url(self, title):
        return "/thumbs/" + os.path.basename(thumb_path("", title))

    def query_random_test(self):
        i = int(np.random.choice(self.test))
        r = self.recs[i]
        img = Image.open(thumb_path(os.path.join(self.data, "thumbs"), r["title"]))
        return self.wrap(img, r, exclude=i, truth=r["categories"])

    def query_commons(self, text):
        title = urllib.parse.unquote(text.strip().split("/wiki/")[-1]).replace("_", " ")
        if not title.lower().startswith("file:"):
            title = "File:" + title
        data = api_get({"action": "query", "titles": title, "prop": "imageinfo|categories",
                        "iiprop": "url|mime|extmetadata", "iiurlwidth": str(THUMB_WIDTH),
                        "iiextmetadatafilter": "ImageDescription", "clshow": "!hidden", "cllimit": "max"})
        page = data["query"]["pages"][0]
        info = (page.get("imageinfo") or [{}])[0]
        if page.get("missing") or not info.get("mime", "").startswith("image/"):
            raise ValueError(f"{title} is not an image on Commons")
        meta = info.get("extmetadata") or {}
        rec = {"title": page["title"], "description": strip_html(meta.get("ImageDescription", {}).get("value"))}
        path = thumb_path(os.path.join(self.data, "thumbs"), rec["title"])
        if not download(info["thumburl"], path):
            raise ValueError("could not download the thumbnail")
        exclude = next((i for i, r in enumerate(self.recs) if r["title"] == rec["title"]), None)
        truth = [c["title"] for c in page.get("categories", [])]
        return self.wrap(Image.open(path), rec, exclude=exclude, truth=truth)

    def query_upload(self, image_b64, title, description):
        raw = base64.b64decode(image_b64.split(",")[-1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((THUMB_WIDTH * 2, THUMB_WIDTH * 2))
        rec = {"title": "File:" + (title.strip() or "Uploaded photo"), "description": description.strip()}
        name = hashlib.md5(raw).hexdigest() + ".jpg"
        img.save(os.path.join(self.data, "thumbs", name), "JPEG", quality=90)
        res = self.wrap(img, rec, exclude=None, truth=[])
        res["query"]["thumb"] = "/thumbs/" + name
        return res

    def wrap(self, img, rec, exclude, truth):
        with self.lock:
            res = self.suggest(img, rec, exclude)
        res["query"] = {"title": rec["title"], "description": rec.get("description", ""),
                        "thumb": self.thumb_url(rec["title"]), "truth": truth}
        return res


def make_handler(sugg):
    page = open(os.path.join(HERE, "demo.html"), "rb").read()

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.send(200, page, "text/html; charset=utf-8")
            m = re.fullmatch(r"/thumbs/([0-9a-f]{32}\.jpg)", self.path)
            if m:
                path = os.path.join(sugg.data, "thumbs", m.group(1))
                if os.path.exists(path):
                    return self.send(200, open(path, "rb").read(), "image/jpeg")
            if self.path == "/api/random":
                return self.api(sugg.query_random_test)
            self.send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
            if self.path == "/api/commons":
                return self.api(lambda: sugg.query_commons(body.get("title", "")))
            if self.path == "/api/upload":
                return self.api(lambda: sugg.query_upload(body["image"], body.get("title", ""),
                                                          body.get("description", "")))
            if self.path == "/api/feedback":
                with sugg.lock:
                    return self.send(200, sugg.feedback(body["id"], body["category"], body["decision"]))
            self.send(404, {"error": "not found"})

        def api(self, fn):
            try:
                self.send(200, fn())
            except Exception as e:
                self.send(400, {"error": str(e)})

        def log_message(self, fmt, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print("Loading model and index...")
    sugg = Suggester(args.data)
    print(f"Ready: {len(sugg.recs)} images, {len(sugg.cats)} categories, on {sugg.device}")
    print(f"Open http://localhost:{args.port}")
    # Listen on both IPv4 and IPv6 localhost: browsers on macOS often try "localhost"
    # over IPv6 (::1) first. Both are loopback only, so nothing is exposed on the network.
    handler = make_handler(sugg)

    class V6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    servers = [ThreadingHTTPServer(("127.0.0.1", args.port), handler)]
    try:
        servers.append(V6Server(("::1", args.port), handler))
    except OSError as e:
        print(f"(IPv6 localhost unavailable: {e}; use http://127.0.0.1:{args.port})")
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
