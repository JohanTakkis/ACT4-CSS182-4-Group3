# ACT4-CSS182-4-Group3 — NLP Model Variants: BERT · GPT-2 · Text-GAN

**Group members:** Alon, John Kenneth · Bunag, Annika · Villafranca, Johan Takkis

Comparative study of three NLP architectures fine-tuned on the IMDb movie review corpus: BERT for sentiment classification, GPT-2 for causal language generation, and a CNN+GRU Text-GAN for adversarial text generation.

## Dataset

- **Source:** [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Splits:** 4,000 train / 1,000 validation / 1,000 test (subsampled from the official 25k/25k IMDb splits)

| Split | Size | Positive | Negative |
|-------|------|----------|----------|
| Train | 4,000 | 1,985 | 2,015 |
| Val   | 1,000 | 521   | 479   |
| Test  | 1,000 | 488   | 512   |

## Results

| Model | Task | Precision | Recall | F1 | BLEU-4 | ROUGE-L | Perplexity | Epoch Time |
|-------|------|-----------|--------|----|--------|---------|------------|------------|
| BERT (bert-base-uncased) | Sentiment Classification | 0.8676 | 0.8675 | 0.8670 | N/A | 0.6714 | N/A | ~58.5s |
| GPT-2 (distilgpt2) | Text Generation | N/A | N/A | N/A | 0.0004 | 0.1162 | 36.56 | ~222.6s |
| Text-GAN (CNN+GRU) | Adversarial Generation | 1.0000* | 1.0000* | 1.0000* | 0.0000 | 0.0100 | N/A | ~1.0s |

\* Text-GAN precision/recall/F1 reflect the **discriminator's** real-vs-fake classification, not generation quality. A perfect discriminator score combined with near-zero generator BLEU/ROUGE is the signature of **mode collapse** — see Part 6.3.3 of the notebook for full analysis.

## How to Run

1. Open `notebooks/NLP_Project_Colab.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Runtime → **Run all**

## Repository Structure

```
ACT4-CSS182-4-Group3/
├── data/
│   └── split_summary.json          # Train/val/test split sizes and class balance
├── models/
│   ├── bert/
│   │   ├── training_history.json   # Final metrics + full classification report
│   │   └── test_inferences.json
│   ├── gpt2/
│   │   ├── training_history.json   # Loss/perplexity/BLEU/ROUGE summary
│   │   └── test_inferences.json    # 5 real prompt → generated-text pairs
│   └── textgan/
│       ├── training_history.json   # D_loss/G_loss/D_acc per epoch (10 epochs)
│       └── test_inferences.json    # 5 real unconditioned generated samples
├── notebooks/
│   └── NLP_Project_Colab.ipynb     # Full executed notebook (Parts 1–7)
└── outputs/
    ├── bert_results.json
    ├── gpt_results.json
    ├── gan_results.json
    └── full_summary.json
```

## Notebook Structure

| Part | Contents |
|------|----------|
| 1 | Environment setup & IMDb dataset loading/splitting |
| 2 | BERT fine-tuning & evaluation |
| 3 | GPT-2 (DistilGPT-2) fine-tuning, perplexity, BLEU/ROUGE |
| 4 | Text-GAN (GRU Generator + 1D-CNN Discriminator) adversarial training |
| 5 | Consolidated performance comparison matrix & visualization |
| 6 | Analytical discussion: tokenization effects & metric tradeoffs |
| 7 | References (13 sources, 2024–2026) |

## Requirements

```
transformers
datasets==2.18.0
numpy<2.0
torch
scikit-learn
matplotlib
```
