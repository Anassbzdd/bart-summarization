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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Wandb
wandb.init(project=WANDB_PROJECT)

# 1. load model
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
# Move model to GPU
model.to(device)

# 2. load datasets
train_data , val_data = get_datasets()

# 3. data collator
data_collator = DataCollatorForSeq2Seq(
    model= model,
    padding= True,
    tokenizer=tokenizer
)

rouge = evaluate.load('rouge')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    decoded_pred = tokenizer.batch_decode(
        logits, 
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

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size= BATCH_SIZE,
    per_device_eval_batch_size= BATCH_SIZE,
    num_train_epochs= NUM_EPOCHS,
    learning_rate= LEARNING_RATE,
    fp16= True,
    logging_steps=LOGGING_STEPS,
    warmup_steps= WARMUP_STEPS,
    weight_decay= WEIGHT_DECAY,
    save_strategy= 'epoch',
    predict_with_generate=True, # every time the model wants to calculate the rote will use this line
    eval_strategy = 'epoch',
    report_to='wandb',
    run_name= 'model',
    load_best_model_at_end=True
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

trainer.save_model('/content/drive/MyDrive/project/best_model')
tokenizer.save_pretrained('/content/drive/MyDrive/project/best_model')



