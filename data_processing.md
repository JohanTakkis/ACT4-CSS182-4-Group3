# Structured Data Processing

This file documents the full data processing pipeline used in `notebooks/NLP_Project_Colab.ipynb` (Part 1), from raw dataset acquisition through to the three split-specific, model-ready artifacts consumed by BERT, GPT-2, and Text-GAN. All steps below are read directly from the executed notebook cells (Part 1.1–1.4) and their saved outputs.

---

## 1. Source Dataset

| Property | Value |
|---|---|
| Dataset | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (Hugging Face Hub) |
| Loader | `datasets.load_dataset("stanfordnlp/imdb")` |
| Native splits | `train` (25,000), `test` (25,000) |
| Task | Binary sentiment classification (0 = negative, 1 = positive) |

The full IMDb dataset contains 50,000 labeled movie reviews. For compute-budget reasons (Colab T4 GPU, fine-tuning scale rather than full-scale pretraining), a fixed subsample was drawn rather than using all 25,000/25,000 examples — see Section 2.

---

## 2. Subsampling and Split Construction (Part 1.4)

```python
TRAIN_SIZE = 4000   # max 25000
VAL_SIZE   = 1000
TEST_SIZE  = 1000
```

| Split | Source | Size | Construction |
|---|---|---|---|
| Train | native `train` | 4,000 | `raw["train"].shuffle(seed=42).select(range(4000))` |
| Val | native `train` | 1,000 | Same shuffled pool, `select(range(4000, 5000))` — disjoint from train |
| Test | native `test` | 1,000 | `raw["test"].shuffle(seed=42).select(range(1000))` — entirely separate native split, never touched during train/val construction |

**Reproducibility:** `seed=42` is fixed for every shuffle call, so the exact same 4,000/1,000/1,000 examples are drawn on every run.

**Actual class balance** (recovered from executed cell output):

| Split | Size | Positive | Negative |
|---|---|---|---|
| Train | 4,000 | 1,985 | 2,015 |
| Val | 1,000 | 521 | 479 |
| Test | 1,000 | 488 | 512 |

---

## 3. Text Cleaning

```python
def clean(text):
    return re.sub(r"<.*?>", " ", text).strip()
```

Applied to every example's `text` field across all three splits. The raw IMDb text contains HTML line-break tags (`<br /><br />`) inherited from the original web-scraped reviews; the regex strips any `<...>` tag pattern (non-greedy match) and replaces it with a single space, then `.strip()` trims the resulting leading/trailing whitespace. No other normalization (lowercasing, punctuation removal, stopword filtering) is applied at this stage — lowercasing and tokenization-specific normalization happen downstream, inside each model's own tokenizer.

---

## 4. Persisted Artifacts

| File | Contents | Consumed by |
|---|---|---|
| `data/train.json` | `{"texts": [...], "labels": [...]}`, 4,000 cleaned reviews | BERT (Part 2.1) |
| `data/val.json` | Same structure, 1,000 reviews | BERT (Part 2.2 evaluation), GPT-2 perplexity (Part 3.3) |
| `data/test.json` | Same structure, 1,000 reviews | BERT (Part 2.3), GPT-2 generation (Part 3.4), Text-GAN evaluation (Part 4.4–4.5) |
| `data/corpus.txt` | All 6,000 cleaned texts (train+val+test) concatenated, one review per line | GPT-2 causal LM training (Part 3.1), Text-GAN vocabulary construction (Part 4.1) |

`split_summary.json` in this repository's `data/` folder records the split sizes and class balance shown above; the raw `train.json` / `val.json` / `test.json` / `corpus.txt` files themselves are produced at runtime inside Colab and are regenerated identically every run due to the fixed seed.

---

## 5. Model-Specific Downstream Processing

Each model branches from the shared `clean()`-ed text/label artifacts above into its own tokenization and encoding pipeline (full detail in `model_pipelines.md`):

| Model | Tokenizer | Vocabulary | Sequence length | Encoding unit |
|---|---|---|---|---|
| BERT | `BertTokenizer` (`bert-base-uncased`, WordPiece) | 30,522 (pretrained) | 128 tokens, padded/truncated | Per-review |
| GPT-2 | `GPT2Tokenizer` (`distilgpt2`, BPE) | 50,257 (pretrained) | 128-token contiguous blocks | Concatenated corpus, chunked |
| Text-GAN | Custom regex word-level (`re.findall(r"\b\w+\b", ...)`) | 5,000 (built from corpus, top-4,997 words + 3 special tokens) | 32 tokens, padded/truncated | Per-review |

The three models therefore see the *same* underlying cleaned text but in three structurally different numeric representations — this divergence is the basis of the tokenization analysis in the notebook's Part 6.2.

---

## 6. Pipeline Diagram

```
stanfordnlp/imdb (Hugging Face Hub)
        │
        ▼
  load_dataset()  →  raw["train"] (25k), raw["test"] (25k)
        │
        ▼
  shuffle(seed=42) + select()  →  4,000 train / 1,000 val / 1,000 test
        │
        ▼
  clean()  →  HTML tags stripped, whitespace trimmed
        │
        ├──────────────┬──────────────────┐
        ▼              ▼                  ▼
  train/val/test    corpus.txt        split_summary.json
  .json (labeled)   (unlabeled,       (sizes + balance)
        │            all 6,000 lines)        
        ▼                  │
  BertTokenizer        ┌───┴────┐
  (WordPiece, 128)     ▼        ▼
        │         GPT2Tokenizer  Custom word-level
        ▼         (BPE, 128-     tokenizer
  BERT model      token blocks)  (5,000 vocab, 32)
                       │              │
                       ▼              ▼
                  GPT-2 model    Text-GAN
                                 (Generator + Discriminator)
```
