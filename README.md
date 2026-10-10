# Automatic Category Suggestion for Wikimedia Commons

## Requirements

- Python 3.10 or newer
- About 1 GB of free disk space for thumbnails and embeddings
- Internet access (Wikimedia Commons API and model download)
- Optional: a CUDA GPU or Apple Silicon (MPS); everything also runs on CPU

## Setup

```bash
git clone https://github.com/IIIT-ECell/megathon26-compileandconquer.git
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

macOS with Python from python.org only: run `Install Certificates.command`
(in `/Applications/Python 3.x/`) once, or HTTPS requests will fail.

Before fetching data, set your own contact email in `USER_AGENT` at the top of
`fetch_commons.py`; Wikimedia requires one.

## Run

Run the steps in order from the repository folder. Each step caches its work,
so an interrupted step can simply be run again.

```bash
# 1. Fetch images and categories from Wikimedia Commons (~10 min)
python3 fetch_commons.py --seeds seed_categories.txt --per-cat 150
# optional: add random Commons images
python3 fetch_commons.py --random 6000

# 2. Download thumbnails and compute embeddings (~20-30 min)
python3 embed.py

# 3. Find candidate categories via Commons search (~30-40 min)
python3 candidates.py

# 4. Evaluate
python3 baseline.py

# Optional: robustness and speed measurements
python3 robustness.py
python3 benchmark.py

# Demo web app, then open http://localhost:8000
python3 demo.py
```

`demo.py` needs steps 1-4 to have been run once. All outputs are written to
`data/` and `commons_data.jsonl`, which are not part of the repository.

## External data and model weights

- **Dataset:** fetched at run time from [Wikimedia Commons](https://commons.wikimedia.org)
  through the [MediaWiki Action API](https://commons.wikimedia.org/w/api.php).
  No dataset is stored in this repository. Images and metadata keep their
  original Commons licences.
- **Model weights:** OpenCLIP ViT-B/32 trained on LAION-2B
  ([laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)),
  downloaded automatically by [open_clip](https://github.com/mlfoundations/open_clip)
  on first run.

## License

[MIT](LICENSE)
