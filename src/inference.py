from transformers import BartForConditionalGeneration , BartTokenizer
import torch.cuda as cuda
import evaluate
import numpy as np
from datasets import load_dataset
from config import *
import torch

device = 'cuda' if cuda.is_available() else 'cpu'

results = {
    '/content/drive/MyDrive/project/best_model': None,
    'facebook/bart-base': None,
    'facebook/bart-large-cnn': None
}

article = """Manchester United are in pole position to secure Champions League qualification after a dramatic reversal in their performances and results since Michael Carrick replaced the fired Ruben Amorim as coach in January.

    United have climbed to third position in the Premier League after taking 19 points from a possible 24 since Carrick took charge, and they face fourth-place Aston Villa at Old Trafford on Sunday knowing that a win will hand them a major boost in their pursuit of a top-five finish.

    But despite the upturn in fortunes since Carrick's arrival, uncertainty remains over the 44-year-old's future beyond the end of his contract in May. United have yet to decide on who will manage the club permanently -- Carrick is among several options -- and moves are also happening behind the scenes to ensure that the club secures its priority transfer candidates this summer, regardless of who is appointed as head coach.

    So what is happening with Carrick and United's recruitment plans? ESPN has spoken to several sources to find out."""

def summarize(article, model, tokenizer):
        input = tokenizer(
            article,
            max_length =1024,
            truncation = True,
            return_tensors = 'pt'
        ).to(device)

        output = model.generate(
            input['input_ids'],
            max_length = 128,
            min_length = 10,
            early_stopping=True
        )

        return tokenizer.decode(
            output[0],
            skip_special_tokens = True
        )

for split in ['/content/drive/MyDrive/project/best_model',
              'facebook/bart-base',
              'facebook/bart-large-cnn']:
    model = BartForConditionalGeneration.from_pretrained(split).to(device)
    tokenizer = BartTokenizer.from_pretrained(split) 
    result = summarize(article, model, tokenizer)
    results[split] = result
    print(result)

dataset = load_dataset(DATASET_NAME,DATASET_VERSION)
data = dataset['validation'].select(range(50))

rouge = evaluate.load('rouge')

for split in ['/content/drive/MyDrive/project/best_model',
              'facebook/bart-base',
              'facebook/bart-large-cnn']:
      
      predictions = []
      references = []

      model = BartForConditionalGeneration.from_pretrained(split).to(device)
      tokenizer = BartTokenizer.from_pretrained(split)
      
      for justone in data:
        article = justone['article'] 
        labels = justone['highlights']
        prediction = summarize(article, model , tokenizer)

        predictions.append(prediction)
        references.append(labels)
        
      metric = rouge.compute(predictions=predictions,
                    references=references)
      print(f'Split => {split}, Metric => {metric}')

      del model
      torch.cuda.empty_cache()

      


            



