# BART Text Summarization — CNN/DailyMail Fine-Tune

Fine-tuning `facebook/bart-base` on the CNN/DailyMail dataset for abstractive text summarization. Trained with [Hugging Face Transformers](https://huggingface.co/docs/transformers) and tracked via [Weights & Biases](https://wandb.ai/).

---

## 📋 Project Overview

| Item | Detail |
|---|---|
| **Base Model** | `facebook/bart-base` |
| **Dataset** | CNN/DailyMail |
| **Task** | Abstractive Summarization |
| **Training Environment** | Kaggle (GPU T4 x2) |
| **Experiment Tracking** | Weights & Biases (W&B) |

---

## 🏋️ Training Results

| Metric | Value |
|---|---|
| **Train Loss** | 1.9233 |
| **Eval Loss** | 2.1319 |
| **Epochs** | 3 |
| **Global Steps** | 3,750 |
| **Total FLOPs** | ~18.3 PFLOPs |
| **Grad Norm** | 4.77 |
| **Final Learning Rate** | 7.39 × 10⁻⁷ |

---

## 📊 ROUGE Scores

| Metric | Score |
|---|---|
| **ROUGE-1** | 0.2361 |
| **ROUGE-2** | 0.0980 |
| **ROUGE-L** | 0.1965 |
| **ROUGE-Lsum** | 0.2180 |

> ROUGE-1 of ~0.24 is a reasonable baseline for `bart-base` fine-tuned on a subset of CNN/DailyMail. `bart-large-cnn` (fully fine-tuned by Meta) typically achieves ROUGE-1 ~0.44.

---

## ⚡ Throughput

| Phase | Samples/sec | Steps/sec | Runtime |
|---|---|---|---|
| **Training** | 15.25 | 1.91 | ~32.8 min |
| **Evaluation** | 8.23 | 1.03 | ~2.0 min |

---

## 🚀 Quick Start

### Install dependencies

```bash
pip install transformers datasets evaluate rouge_score wandb
```


