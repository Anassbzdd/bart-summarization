import argparse
import torch
from config import *
from preprocessing import get_datasets, get_tokenizer
from transformers import BartForConditionalGeneration , Seq2SeqTrainer , Seq2SeqTrainingArguments , DataCollatorForSeq2Seq 
import evaluate
import numpy as np
import os
import json

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
    return parser.parse_args()

rouge = evaluate.load('rouge')

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

def build_trainer(args , trial_config = None):
    config = trial_config or args
    tokenizer = get_tokenizer(args.model_name)
    train_data , val_data = get_datasets(
        tokenizer= tokenizer,
        train_size=args.train_size,
        val_size=args.val_size ,
    )

    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(
        model = model,
        tokenizer=tokenizer,
        padding= True,
    )

    report_to = 'wandb' if args.use_wandb else None
    run_name = args.run_name if args.use_wandb else None

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
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
        model = model,
        args = training_arguments,
        train_dataset=train_data,
        eval_dataset= val_data,
        data_collator=data_collator,
        tokenizer = tokenizer,
        compute_metrics=compute_metrics_builder(tokenizer)
    )
    return trainer , tokenizer

def save_metrics(metrics, output_path):
    if not output_path:
        return
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)




    