# Model Pipelines

This file documents the architecture, hyperparameters, and training pipeline for each of the three model variants in `notebooks/NLP_Project_Colab.ipynb`. All configuration values are read directly from the executed notebook cells (Parts 2, 3, and 4).

---

## 1. BERT — Sentiment Classification (Part 2)

### Pipeline

```
data/train.json, data/val.json, data/test.json
        │
        ▼
BertTokenizer.from_pretrained("bert-base-uncased")
        │  padding="max_length", truncation=True, max_length=128
        ▼
HFDataset.map(tokenize_fn, batched=True)
        │  .set_format("torch", columns=["input_ids","attention_mask","labels"])
        ▼
BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        │
        ▼
Trainer.train()  →  Trainer.predict(test_ds)
        │
        ▼
outputs/bert_results.json, models/bert/training_history.json
```

### Configuration

| Parameter | Value |
|---|---|
| Base checkpoint | `bert-base-uncased` (110M params) |
| Tokenizer | WordPiece, vocab size 30,522 |
| Max sequence length | 128 tokens |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Epochs | 3 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Eval/save strategy | per epoch |
| `load_best_model_at_end` | `True`, selected by F1 |
| Mixed precision | `fp16=True` (GPU only) |

### Actual results (test set, n=1,000)

| Metric | Value |
|---|---|
| Precision (macro) | 0.8676 |
| Recall (macro) | 0.8675 |
| F1 (macro) | 0.8670 |
| Accuracy | 0.8670 |
| Avg epoch time | 58.5s |

---

## 2. GPT-2 (`distilgpt2`) — Causal Language Modeling (Part 3)

### Pipeline

```
data/corpus.txt  (6,000 lines, all splits concatenated)
        │
        ▼
GPT2Tokenizer.from_pretrained("distilgpt2")
        │  pad_token = eos_token
        ▼
CorpusDataset  →  tokenize full corpus, chunk into 128-token blocks
        │
        ▼
DataCollatorForLanguageModeling(mlm=False)
        │
        ▼
GPT2LMHeadModel.from_pretrained("distilgpt2")
        │
        ▼
Trainer.train()
        │
        ├──→ compute_perplexity(model, val_texts[:100])
        │
        └──→ pipeline("text-generation") for qualitative + BLEU/ROUGE samples
        │
        ▼
outputs/gpt_results.json, models/gpt2/training_history.json,
models/gpt2/test_inferences.json
```

### Configuration

| Parameter | Value |
|---|---|
| Base checkpoint | `distilgpt2` (82M params) |
| Tokenizer | BPE, vocab size 50,257 |
| Block size | 128 tokens (contiguous, non-overlapping) |
| Batch size | 8 |
| Epochs | 5 |
| Generation sampling | `do_sample=True, temperature=0.8, top_p=0.92` |
| Mixed precision | `fp16=True` (GPU only) |

### Actual results

| Metric | Value |
|---|---|
| Validation perplexity | 36.56 |
| BLEU-4 | 0.0004 |
| ROUGE-L | 0.1162 |
| Total training time | 1113.0s |
| Avg epoch time | 222.6s |

---

## 3. Text-GAN — Adversarial Text Generation (Part 4)

### Pipeline

```
data/corpus.txt (train_texts only)
        │
        ▼
Custom vocabulary: re.findall(r"\b\w+\b", ...) over full corpus
        │  top 4,997 words + <PAD>, <UNK>, <EOS>  →  vocab_size = 5,000
        ▼
encode(text) → fixed 32-token integer ID sequences (SeqDataset)
        │
        ▼
DataLoader(batch_size=32, shuffle=True, drop_last=True)
        │
        ├──────────────────────┬─────────────────────────┐
        ▼                      ▼
   Generator (GRU)        Discriminator (1D-CNN)
        │                      │
        └─────── adversarial training loop (10 epochs) ───────┘
        │
        ├──→ Discriminator eval: 500 real vs. 500 generated, held-out
        └──→ Generator eval: 30 generated samples, BLEU-4/ROUGE-L vs. real test text
        │
        ▼
outputs/gan_results.json, models/textgan/training_history.json,
models/textgan/test_inferences.json
```

### Generator architecture

```python
class Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len):
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru    = nn.GRU(embed_dim, hidden_dim, num_layers=2,
                              batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_dim, vocab_size)
```

| Layer | Spec |
|---|---|
| Embedding | vocab_size=5,000 → embed_dim=64, `padding_idx=0` |
| GRU | 2 layers, hidden_dim=128, `dropout=0.3` between layers |
| Output linear | hidden_dim=128 → vocab_size=5,000 logits |
| Generation method | `argmax` over logits from random-noise input (no prompt conditioning) |

### Discriminator architecture

```python
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, seq_len):
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(num_filters, 1)
```

| Layer | Spec |
|---|---|
| Embedding | vocab_size=5,000 → embed_dim=64, `padding_idx=0` |
| Conv1d (×2) | 64→128 channels (kernel=3, pad=1), then 128→128 (kernel=5, pad=2), ReLU after each |
| Pooling | `AdaptiveMaxPool1d(1)` — global max over the sequence dimension |
| Dropout | 0.3 |
| Output linear | 128 → 1, followed by sigmoid (real/fake probability) |

### Training configuration

| Parameter | Value |
|---|---|
| Sequence length | 32 tokens |
| Batch size | 32 |
| Epochs | 10 |
| Generator optimizer | Adam, `lr=1e-3`, `betas=(0.5, 0.999)` |
| Discriminator optimizer | Adam, `lr=1e-4`, `betas=(0.5, 0.999)` |
| Loss | `nn.BCELoss()` |
| Generator gradient clipping | `clip_grad_norm_(G.parameters(), 1.0)` |
| Pretraining | **None** — both networks trained fully from random initialization |

### Actual results

| Metric | Value |
|---|---|
| Discriminator accuracy (held-out) | 1.0000 |
| Discriminator precision/recall/F1 (macro) | 1.0000 / 1.0000 / 1.0000 |
| Generator BLEU-4 | 0.0000 |
| Generator ROUGE-L | 0.0100 |
| Total training time | ~10.0s (10 epochs) |
| Avg epoch time | 1.0s |

See `training_history.md` for the full per-epoch loss curve and a discussion of why this discriminator-dominant pattern indicates mode collapse rather than successful adversarial convergence.

---

## 4. Cross-Model Comparison Summary

| Property | BERT | GPT-2 | Text-GAN |
|---|---|---|---|
| Task | Classification | Generation | Adversarial generation |
| Pretrained | Yes (110M params) | Yes (82M params) | No (random init) |
| Tokenizer | WordPiece (30,522) | BPE (50,257) | Custom word-level (5,000) |
| Sequence length | 128 | 128-token blocks | 32 |
| Epochs | 3 | 5 | 10 |
| Avg epoch time | 58.5s | 222.6s | 1.0s |
| Primary metric | F1 = 0.8670 | Perplexity = 36.56 | Discriminator Acc. = 1.0000 |
| Generative quality | N/A | BLEU-4 = 0.0004, ROUGE-L = 0.1162 | BLEU-4 = 0.0000, ROUGE-L = 0.0100 |
