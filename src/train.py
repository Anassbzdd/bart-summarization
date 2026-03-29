import argparse
import json
import os

import evaluate
import numpy as np
import torch
import wandb
from transformers import (
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOGGING_STEPS,
    MODEL_NAME,
    NUM_EPOCHS,
    OUTPUT_DIR,
    WANDB_PROJECT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
)
from preprocessing import get_datasets, get_tokenizer


rouge = evaluate.load("rouge")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BART on CNN/DailyMail.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-metrics-path", default=None)
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--sweep-count", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--early-stopping-threshold", type=float, default=EARLY_STOPPING_THRESHOLD)
    return parser.parse_args()


def compute_metrics_builder(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        return rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )

    return compute_metrics


def build_trainer(args, trial_config=None):
    config = trial_config or args
    tokenizer = get_tokenizer(args.model_name)
    train_data, val_data = get_datasets(
        tokenizer=tokenizer,
        train_size=args.train_size,
        val_size=args.val_size,
    )

    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    data_collator = DataCollatorForSeq2Seq(
        model=model,
        tokenizer=tokenizer,
        padding=True,
    )

    report_to = "wandb" if args.use_wandb else "none"
    run_name = args.run_name if args.use_wandb else None

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        save_total_limit=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.generation_max_length,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),
        report_to=report_to,
        run_name=run_name,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics_builder(tokenizer),
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )
    return trainer, tokenizer


def save_metrics(metrics, output_path):
    if not output_path:
        return
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def train_once(args):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name)

    trainer, tokenizer = build_trainer(args)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "train": train_result.metrics,
        "eval": eval_metrics,
    }
    save_metrics(metrics, args.save_metrics_path)

    if args.use_wandb:
        wandb.log(eval_metrics)
        wandb.finish()

    return metrics


def build_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/rougeL",
            "goal": "maximize",
        },
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-4,
            },
            "weight_decay": {
                "values": [0.01, 0.05, 0.1],
            },
            "batch_size": {
                "values": [4, 8],
            },
            "num_epochs": {
                "values": [3, 5],
            },
            "warmup_steps": {
                "values": [100, 300, 500],
            },
        },
    }


def run_sweep(args):
    sweep_id = wandb.sweep(build_sweep_config(), project=args.wandb_project)

    def train_sweep():
        with wandb.init(project=args.wandb_project):
            trainer, _ = build_trainer(args, trial_config=wandb.config)
            trainer.train()
            metrics = trainer.evaluate()
            wandb.log(metrics)

    wandb.agent(sweep_id, function=train_sweep, count=args.sweep_count)


def main():
    args = parse_args()
    if args.run_sweep:
        if not args.use_wandb:
            raise ValueError("--run-sweep requires --use-wandb.")
        run_sweep(args)
        return

    metrics = train_once(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
