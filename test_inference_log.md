# Test Inference Log

Qualitative inference outputs from all three model variants, captured from the executed training run documented in `notebooks/NLP_Project_Colab.ipynb`. This is a standalone extract — the full code that produced these outputs lives in the notebook's Part 2.3 (BERT evaluation), Part 3.4 (GPT-2 generation & BLEU/ROUGE), and Part 4.4–4.5 (Text-GAN discriminator and generator evaluation) cells.

---

## 1. BERT — Classification (quantitative, not generative)

BERT has no generation capability — its "inference" is a classification decision (positive/negative sentiment), not free text. Its qualitative output is therefore reported via metrics rather than sample text:

| Metric | Value |
|---|---|
| Precision (macro) | 0.8676 |
| Recall (macro) | 0.8675 |
| F1 (macro) | 0.8670 |
| Accuracy | 0.8670 |
| ROUGE-L (rationale spans) | 0.6714 |

Per-class breakdown (test set, n=1000):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.8891 | 0.8457 | 0.8669 | 512 |
| Positive | 0.8460 | 0.8893 | 0.8671 | 488 |

---

## 2. GPT-2 (`distilgpt2`) — Generative Inference

### 2a. Domain-specific prompt completion (Part 3.4)

Five fixed prompts, each completed with `do_sample=True, temperature=0.8, top_p=0.92, max_new_tokens=30`.

| # | Prompt | Generated continuation |
|---|---|---|
| 1 | "This film is a masterpiece" | "This film is a masterpiece, and one that I don't expect to see again. I have to admit that I am not quite sure I could stomach this film, as..." |
| 2 | "The director completely failed" | "The director completely failed to take a risk in his direction, and I think his efforts should have been avoided. I think the film is an attempt at a more sophisticated approach..." |
| 3 | "A heartwarming story about" | "A heartwarming story about the people who are destined to make love. The film takes place in a small town in the late 60s and early 70s. The movie features..." |
| 4 | "I was deeply disappointed by" | "I was deeply disappointed by the film. In one of the most entertaining, moving moments of the film, the police officer's character's wife (played by Alan Arkin)..." |
| 5 | "The cinematography and script" | "The cinematography and script is excellent, especially the opening scene where the two lovers are separated by a fence and it looks like someone has been killed. The ending is an interesting..." |

### 2b. Prompt → ground-truth-continuation comparisons, with BLEU/ROUGE (Part 3.4)

30 real test-set sentences, each split into a 5-token prompt; the model's own continuation from that prompt is compared against the actual remainder of the sentence.

**Aggregate (30 scored test samples):** BLEU-4 = 0.0004, ROUGE-L = 0.1162, Perplexity (val) = 36.56.

*Note on low BLEU despite fluent output:* BLEU/ROUGE penalize any lexical divergence from one specific reference continuation. The generated text above is grammatical and topically on-point, but uses different words/phrasing than the one ground-truth continuation it's scored against — this is expected and is discussed further in the notebook's Part 6 analytical write-up (Section 6.3.2).

---

## 3. Text-GAN (1D-CNN Discriminator + GRU Generator, word-level) — Generative Inference

### 3a. Decoded generator samples (Part 4.5 evaluation, unconditioned — no prompt)

| # | Decoded output |
|---|---|
| 1 | "incredible tad remember mainly sits sits eric eric eric connection connection answers acting brosnan forms forms creatures creatures creatures creatures awe front european didn didn didn didn movement movement movement sitcom potential" |
| 2 | "mainly cookie access access wanting this ship talk access war beginning view european snake cookie gina talk wanting pleasure pleasure pleasure horror horror greek sitcom jungle receive suspenseful we lyrics standout standout" |
| 3 | "mountain access empty acting acting acting roles empty authenticity forms bottom solid king king king sitcom count gina sitcom sitcom sitcom sitcom mountain we we we this blues persons sitcom wanting blues" |
| 4 | "united mountain solve tremendous worse rip rip creatures answers adams environment this source source partly count count count count count we we eventually eventually foster foster originally soviet a a deep doc" |
| 5 | "rip rip rip answers answers tends tends dangerous attached overall dangerous disappointing disappointing producer skits blues european dangerous wished wished this philosophical uninteresting european same same spots task task molly didn returns" |

**Aggregate (30 generated samples, compared against the first 30 real test-set sentences):** BLEU-4 = 0.0000, ROUGE-L = 0.0100.

**Discriminator classification (500 real vs. 500 generated, held-out):** Accuracy = 1.0000, Precision (macro) = 1.0000, Recall (macro) = 1.0000, F1 (macro) = 1.0000.

*Why the output looks like this:* this run's training history (see `training_history.md`) shows the discriminator pulling ahead of the generator from epoch 1 onward, with **no recovery phase at any point** across all 10 epochs — discriminator accuracy crosses 0.99 by epoch 3 and is essentially saturated (0.9999) from epoch 7 through epoch 10. By the final epoch, the discriminator achieves a perfect 1.0000 accuracy on the held-out evaluation set, meaning `D(G(z))` is driven toward 0 for nearly every generated sample, sharply weakening the gradient signal reaching the generator throughout training. The repeated cycling through a small set of specific words ("masterpiece," "host," "nettelbeck," "creatures," "sitcom") rather than diverse, grammatical sentences is the visible symptom of that imbalance. Full diagnosis in the notebook's Part 6.3.3.

---

## Side-by-side: GPT-2 vs. Text-GAN on the same task

Both models are asked to produce free-form text without a paired ground-truth target (Part 4.5 / 3.4's open generation). The contrast is stark:

- **GPT-2:** fluent, grammatical, topically coherent multi-clause sentences.
- **Text-GAN:** long runs of the same handful of nouns/adjectives repeated in different orders, with no real grammatical structure.

This is the empirical core of the notebook's Part 6 discussion: autoregressive transformer-based generation (GPT-2, with large-scale pretraining and dense per-token gradient signal) versus adversarially-trained discrete-token generation (Text-GAN, trained fully from scratch with sparse, non-differentiable gradient signal) under a comparable fine-tuning-scale compute budget, and what a real, one-sided, discriminator-dominant GAN training curve looks like in practice.
