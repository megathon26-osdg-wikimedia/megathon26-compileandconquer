"""Fetch images + their categories from Wikimedia Commons into a JSONL file.

Usage:
    python3 fetch_commons.py "Category:Bridges in Paris" "Category:Sunsets" --per-cat 50
    python3 fetch_commons.py --seeds seed_categories.txt --per-cat 150

Each output line: {title, timestamp, uploader, thumb_url, description, categories,
seed, seed_type}
"""
import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia requires a descriptive User-Agent with contact info, or requests get blocked.
USER_AGENT = "MegathonCategorySuggester/0.1 (student hackathon project; bhakti.agrawal@research.iiit.ac.in)"
THUMB_WIDTH = 330  # a standard Commons thumb size (others get rounded up); CLIP uses 224px anyway


def api_get(params):
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(5 * (attempt + 1))  # network blip, e.g. after laptop sleep
            continue
        if data.get("error", {}).get("code") == "maxlag":
            time.sleep(5 * (attempt + 1))
            continue
        if "error" in data:
            raise RuntimeError(data["error"])
        return data
    raise RuntimeError("API request kept failing (network errors or maxlag)")


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def fetch_category(category, limit):
    """Yield file records from one category, following API continuation."""
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": "50",
        "prop": "imageinfo|categories",
        "iiprop": "url|timestamp|user|mime|extmetadata",
        "iiurlwidth": str(THUMB_WIDTH),
        "iiextmetadatafilter": "ImageDescription",
        "clshow": "!hidden",  # drop hidden/maintenance categories
        "cllimit": "max",
    }
    count = 0
    while True:
        data = api_get(params)
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("mime", "").startswith("image/"):
                continue  # skip audio, video, PDFs
            meta = info.get("extmetadata") or {}  # API sends [] instead of {} when empty
            desc = meta.get("ImageDescription", {}).get("value")
            yield {
                "title": page["title"],
                "timestamp": info.get("timestamp"),
                "uploader": info.get("user"),
                "thumb_url": info.get("thumburl"),
                "description": strip_html(desc),
                "categories": [c["title"] for c in page.get("categories", [])],
            }
            count += 1
            if count >= limit:
                return
        if "continue" not in data:
            return
        params.update(data["continue"])
        time.sleep(0.5)  # be polite


def read_seeds(path):
    """Return [(category, type)] from a seed file; "# type" lines set the type."""
    seeds, kind = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#"):
            word = line[1:].strip()
            if word and " " not in word:
                kind = word
        elif line:
            seeds.append((line, kind))
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="*")
    ap.add_argument("--seeds", help="file of categories, one per line (see seed_categories.txt)")
    ap.add_argument("--per-cat", type=int, default=100)
    ap.add_argument("--out", default="commons_data.jsonl")
    args = ap.parse_args()

    seeds = [(c, None) for c in args.categories]
    if args.seeds:
        seeds += read_seeds(args.seeds)
    if not seeds:
        ap.error("give categories or --seeds")

    seen = set()
    with open(args.out, "a", encoding="utf-8") as f:
        for cat, kind in seeds:
            n = 0
            for rec in fetch_category(cat, args.per_cat):
                if rec["title"] in seen:
                    continue
                seen.add(rec["title"])
                rec["seed"], rec["seed_type"] = cat, kind
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            print(f"{cat}: {n} files")


if __name__ == "__main__":
    main()
