from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq
)
from config import *
import evaluate
from preprocessing import get_datasets , tokenizer
import numpy as np
import wandb
import os

import torch

sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'eval/rougeL', 
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-4
        },
        'weight_decay': {
            'values': [0.01, 0.05, 0.1]
        },
        'per_device_train_batch_size': {
            'values': [4, 8] 
        },
        'num_train_epochs': {
            'values': [3, 5]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 2. load datasets
train_data , val_data = get_datasets()

rouge = evaluate.load('rouge')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100 ,labels, tokenizer.pad_token_id )
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)

    decoded_pred = tokenizer.batch_decode(
        predictions, 
        skip_special_tokens = True
    )

    labels = np.where(labels != -100 ,labels, tokenizer.pad_token_id )

    decode_target = tokenizer.batch_decode(
        labels,
        skip_special_tokens = True
    )
    rouge_compute = rouge.compute(
        predictions=decoded_pred,
        references=decode_target
    )
    return rouge_compute


def train_sweep():
    with wandb.init():
        config = wandb.config

        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to(device)
        data_collator = DataCollatorForSeq2Seq(
            model= model,
            padding= True,
            tokenizer=tokenizer
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir= f'/kaggle/working/project/checkpoints/{wandb.run.name}',
            per_device_train_batch_size= config.per_device_train_batch_size,
            per_device_eval_batch_size= config.per_device_train_batch_size,
            num_train_epochs= config.num_train_epochs,
            learning_rate= config.learning_rate,
            fp16= torch.cuda.is_available(),
            logging_steps=LOGGING_STEPS,
            warmup_steps= WARMUP_STEPS,
            weight_decay= config.weight_decay,
            save_strategy= 'epoch',
            eval_strategy = 'epoch',
            report_to='wandb',
            load_best_model_at_end=True,
            metric_for_best_model='rougeL',
            run_name = wandb.run.name,
            generation_max_length=128,
            predict_with_generate=True,
            seed = 69

        )

        trainer = Seq2SeqTrainer(
            model= model,
            args = training_args,
            data_collator=data_collator,
            train_dataset= train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
            processing_class= tokenizer
        )

        trainer.train()

        save = f"/kaggle/working/project/{wandb.run.name}"
        trainer.save_model(save)
        tokenizer.save_pretrained(save)
    
wandb.agent(sweep_id, function=train_sweep, count = 6)









