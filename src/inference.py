import argparse

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from config import *


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a summary from input text.")
    parser.add_argument("--model-path", default=MODEL_NAME)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--max-input-length", type=int, default=MAX_INPUT_LENGTH)
    parser.add_argument("--max-summary-length", type=int, default=MAX_TARGET_LENGTH)
    parser.add_argument("--min-summary-length", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=4)
    return parser.parse_args()


def load_article(args):
    if args.text:
        return args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as file:
            return file.read()
    raise ValueError("Provide either --text or --text-file.")


def summarize(article, model, tokenizer, device, args):
    model_inputs = tokenizer(
        article,
        max_length=args.max_input_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    output = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_length=args.max_summary_length,
        min_length=args.min_summary_length,
        num_beams=args.num_beams,
        early_stopping=True,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_model_and_tokenizer(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    return model , tokenizer, device 


def main():
    args = parse_args()
    article = load_article(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BartForConditionalGeneration.from_pretrained(args.model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained(args.model_path)

    summary = summarize(article, model, tokenizer, device, args)
    print(summary)


if __name__ == "__main__":
    main()