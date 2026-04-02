# BART Abstractive Summarization · CNN/DailyMail

Fine-tuning `facebook/bart-base` for abstractive text summarization on CNN/DailyMail, with full experiment tracking via Weights & Biases and a clean CLI for training, inference, and evaluation.

---

## Results

Best checkpoint selected at **epoch 2** based on peak `eval/rouge2`.

| Metric | Score |
|---|---|
| **ROUGE-1** | **0.3792** |
| **ROUGE-2** | **0.1613** |
| **ROUGE-L** | **0.2599** |
| **ROUGE-Lsum** | **0.3527** |

> **Context:** `bart-large` fine-tuned on the *full* CNN/DailyMail dataset (~300k examples) typically reaches ROUGE-2 ~0.21. These results use `bart-base` on 10% of that data (30k examples), making the gap smaller than expected and a strong result for the resource budget used.

---

## Training Setup

| Item | Detail |
|---|---|
| Base model | `facebook/bart-base` (139M params) |
| Dataset | `abisee/cnn_dailymail` 3.0.0 |
| Task | Abstractive summarization |
| Train examples | 30,000 |
| Validation examples | 3,000 |
| Epochs | 5 planned — best checkpoint at epoch 2 |
| Hardware | Kaggle T4 x2 |
| Experiment tracking | Weights & Biases (Bayesian sweep) |

---

## Why This Project

This project covers a complete production-style NLP pipeline:

- **Transfer learning** — adapting a pretrained seq2seq model (BART) to a downstream summarization task
- **Scale** — training on 30k examples with multi-GPU support on Kaggle T4 x2
- **Experiment tracking** — full W&B integration with Bayesian hyperparameter sweeps
- **Evaluation** — ROUGE-1/2/L/Lsum on a held-out validation set with best-checkpoint selection
- **Deployable** — FastAPI wrapper + Dockerfile for serving the model

---

## Project Structure

```text
src/
  config.py           # Central hyperparameter configuration
  preprocessing.py    # Dataset loading, filtering, tokenization
  train.py            # Training entry point with W&B support
  inference.py        # CLI summarization from text or file
  evaluate_model.py   # Held-out evaluation with ROUGE metrics
```

---

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py --output-dir outputs/bart-cnn
```

With W&B logging:

```bash
python src/train.py --output-dir outputs/bart-cnn --use-wandb --run-name bart-run
```

With Bayesian hyperparameter sweep:

```bash
python src/train.py --use-wandb --run-sweep --wandb-project summarization-bart
```

### Inference

From raw text:

```bash
python src/inference.py --model-path outputs/bart-cnn \
  --text "Paste your article here."
```

From a file:

```bash
python src/inference.py --model-path outputs/bart-cnn \
  --text-file sample_article.txt
```

### Evaluate on held-out test set

```bash
python src/evaluate_model.py \
  --model-path outputs/bart-cnn \
  --test-size 100 \
  --save-path outputs/bart-cnn/test_metrics.json
```

---

## License

MIT — see [LICENSE](LICENSE).
