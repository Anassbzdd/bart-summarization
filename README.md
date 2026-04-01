# BART Text Summarization on CNN/DailyMail

This project fine-tunes `facebook/bart-base` for abstractive summarization on the CNN/DailyMail dataset using Hugging Face Transformers. It was built as a portfolio NLP project and includes training, inference, and ROUGE-based evaluation scripts.

## Why This Project Matters

This is a good end-to-end ML example because it covers:

- transfer learning with a pretrained sequence-to-sequence model
- dataset filtering and preprocessing
- experiment tracking with Weights & Biases
- command-line training and inference workflows
- held-out evaluation with ROUGE metrics

## Project Structure

```text
src/
  config.py           # Central configuration values
  preprocessing.py    # Dataset loading, filtering, tokenization
  train.py            # Training entry point
  inference.py        # Summarize custom text from the CLI
  evaluate_model.py   # Evaluate a saved model on held-out test data
```

## Training Setup

| Item | Detail |
|---|---|
| Base model | `facebook/bart-base` |
| Dataset | `abisee/cnn_dailymail` (`3.0.0`) |
| Task | Abstractive summarization |
| Default train subset | 3,000 examples |
| Default validation subset | 300 examples |
| Default test subset | 3 examples |
| Frameworks | Transformers, Datasets, PyTorch |
| Tracking | Weights & Biases |

## Results Snapshot

These are the current reported results from the original run:

| Metric | Value |
|---|---|
| Train loss | 1.9233 |
| Eval loss | 2.1319 |
| ROUGE-1 | 0.2361 |
| ROUGE-2 | 0.0980 |
| ROUGE-L | 0.1965 |
| ROUGE-Lsum | 0.2180 |

Because this project trains on a subset of CNN/DailyMail, the scores should be treated as a baseline rather than a fully optimized benchmark result.

## Installation

```bash
pip install -r requirements.txt
```

## How To Run

Run commands from the project root.

### 1. Train the model

```bash
python src/train.py --output-dir outputs/bart-cnn
```

Optional W&B logging:

```bash
python src/train.py --output-dir outputs/bart-cnn --use-wandb --run-name bart-baseline
```

Optional hyperparameter sweep:

```bash
python src/train.py --use-wandb --run-sweep --wandb-project summarization-bart
```

### 2. Summarize your own text

```bash
python src/inference.py --model-path outputs/bart-cnn --text "Your article goes here."
```

Or summarize from a file:

```bash
python src/inference.py --model-path outputs/bart-cnn --text-file sample_article.txt
```

### 3. Evaluate on held-out test data

```bash
python src/evaluate_model.py --model-path outputs/bart-cnn --test-size 100 --save-path outputs/bart-cnn/test_metrics.json
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
