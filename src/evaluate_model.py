import argparse
from config import MODEL_NAME , TEST_SIZE, MAX_INPUT_LENGTH, MAX_SUMMARY_LEN, MIN_SUMMARY_LEN, DATASET_NAME, DATASET_VERSION
from datasets import load_dataset
import evaluate
from transformers import BartForConditionalGeneration, BartTokenizer
import os
import json
import torch
from preprocessing import _filter_example

def parse_args():
    parser = argparse.ArgumentParser(description = 'Evaluate a summarization model with ROUGE.')
    parser.add_argument('--model-path', default= MODEL_NAME)
    parser.add_argument('--test-size',type=int, default= TEST_SIZE)
    parser.add_argument('--max-input-length',type=int, default= MAX_INPUT_LENGTH)
    parser.add_argument('--max-summary-length',type=int, default= MAX_SUMMARY_LEN)
    parser.add_argument('--min-summary-length',type=int, default= MIN_SUMMARY_LEN)
    parser.add_argument('--save-path',default= None)
    parser.add_argument('--num-beams',type=int,default= 4)
    return parser.parse_args()

def summarize(model, article , tokenizer, device, args):
    input = tokenizer(
        article,
        max_length = args.max_input_length,
        truncation = True,
        return_tensors = 'pt'
    ).to(device)
    output = model.generate(
        input['input_ids'],
        max_length = args.max_summary_length,
        min_length = args.min_summary_length,
        attention_mask = input['attention_mask'],
        num_beams = args.num_beams
    )
    return tokenizer.decode(output[0], skip_special_tokens = True)

def load_test_data(test_size):
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION)
    filtred = dataset['test'].filter(_filter_example)
    return filtred.select(range(min(test_size, len(filtred))))

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rouge = evaluate.load('rouge')
    test_data = load_test_data(args.test_size)

    model = BartForConditionalGeneration.from_pretrained(args.model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained(args.model_path)

    predictions = []
    references = []

    for example in test_data:
        predictions.append(summarize(model, example['article'] , tokenizer, device, args))
        references.append(example['highlights'])
    
    metric = rouge.compute(predictions=predictions, references=references, use_stemmer = True)
    output = {
        'model path':args.model_path,
        'test examples': len(test_data),
        'metrics': metric,
        'samples' : [
            {
                'article' : test_data[i]['article'], 
                'prediction': predictions[i],
                'references': references[i]
            }
            for i in range(min(3, len(test_data)))
        ]
    }

    if args.save_path:
        directory = os.path.dirname(args.save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()





