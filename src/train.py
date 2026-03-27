import argparse
import torch
from config import *
from preprocessing import get_datasets, get_tokenizer
from transformers import BartForConditionalGeneration , Seq2SeqTrainer , Seq2SeqTrainingArguments , DataCollatorForSeq2Seq 

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

def build_trainer():
    args = parse_args()
    tokenizer = get_tokenizer(args.model_name)
    train_data , val_data = get_datasets(tokenizer,
                           train_size= args.train_size ,
                           val_size= args.val_size)
    
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    data_collator = DataCollatorForSeq2Seq(
        model = model,
        tokenizer=tokenizer,
        padding = True
    )


    