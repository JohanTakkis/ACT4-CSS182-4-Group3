# Training History

This file's companion `training_history_summary.csv` captures the **full per-epoch training history** for all three model variants, from the executed run documented in `notebooks/NLP_Project_Colab.ipynb`. All values below are read directly from that notebook's saved cell outputs (Parts 2.2, 3.2/3.3, and 4.3).

## BERT (`bert-base-uncased`, 3 epochs)

| Epoch | Precision | Recall | F1 |
|---|---|---|---|
| 1–2 | — (per-epoch breakdown not separately logged in saved cell output) | — | — |
| final (test set) | 0.8676 | 0.8675 | 0.8670 |

Per-class breakdown (test set, n=1000):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.8891 | 0.8457 | 0.8669 | 512 |
| Positive | 0.8460 | 0.8893 | 0.8671 | 488 |

Total training time: 175.5s → **58.5s / epoch** (3 epochs). Accuracy: 0.8670. ROUGE-L (rationale spans): 0.6714.

## GPT-2 (`distilgpt2`, 5 epochs)

Total training time: **1113.0s** → **222.6s / epoch**. Final perplexity (validation): **36.56**.

| Metric | Value |
|---|---|
| BLEU-4 | 0.0004 |
| ROUGE-L | 0.1162 |
| Perplexity (val) | 36.56 |

*Note on low BLEU despite fluent output:* BLEU/ROUGE penalize any lexical divergence from one specific reference continuation. GPT-2's generated samples (see `test_inference_log.md`) are grammatical and topically on-point, but use different wording than the one ground-truth continuation each is scored against — this is expected and is discussed further in the notebook's Part 6 analytical write-up.

## Text-GAN (1D-CNN Discriminator + GRU Generator, 10 epochs, trained fully from scratch — no pre-trained weights exist for this architecture)

| Epoch | D loss | G loss | D acc | Time (s) |
|---|---|---|---|---|
| 1 | 0.8647 | 1.2574 | 0.8984 | 2.10 |
| 2 | 0.1341 | 3.7429 | 0.9960 | 1.10 |
| 3 | 0.0452 | 5.6293 | 0.9986 | 0.90 |
| 4 | 0.0223 | 6.7706 | 0.9992 | 0.90 |
| 5 | 0.0149 | 7.8727 | 0.9992 | 0.90 |
| 6 | 0.0105 | 8.6345 | 0.9995 | 0.90 |
| 7 | 0.0072 | 9.4537 | 0.9999 | 0.90 |
| 8 | 0.0060 | 10.1737 | 0.9999 | 0.90 |
| 9 | 0.0052 | 10.6960 | 0.9999 | 0.90 |
| 10 | 0.0041 | 10.9807 | 0.9999 | 0.90 |

Average: **1.0s / epoch**. Final held-out evaluation (500 real vs. 500 generated): Discriminator Accuracy=1.0000, Precision=1.0000, Recall=1.0000, F1=1.0000; Generator BLEU-4=0.0000, ROUGE-L=0.0100.

**What the curve shows:** discriminator loss collapses sharply and monotonically from epoch 1 (0.8647) to epoch 10 (0.0041), while generator loss climbs continuously and without interruption from 1.2574 to 10.9807 — a near-eightfold increase. Unlike a healthy adversarial game, where generator and discriminator losses oscillate as each network periodically gains ground, this run shows **no generator recovery at any point** across all 10 epochs: discriminator accuracy crosses 0.99 by epoch 3 and is essentially saturated (0.9999) by epoch 7 onward. This one-sided, ever-widening gap is the signature of **discriminator overpowering / mode collapse**, not stable convergence — the generator received a progressively weaker and less informative gradient signal as training proceeded, since a near-perfect discriminator outputs predictions close to 0 for nearly every generated sample regardless of its content. The repeated cycling through a small set of specific words (see `test_inference_log.md`) rather than diverse, grammatical sentences is the visible symptom of that imbalance. Full diagnosis in the notebook's Part 6.3.3.

## Reproducing this from scratch

1. Open `notebooks/NLP_Project_Colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**.
4. Each variant has no pre-trained checkpoint reload logic — every cell trains the model from its base pretrained weights (BERT, GPT-2) or from full random initialization (Text-GAN), printing its complete per-epoch table/loss log exactly as captured above.
