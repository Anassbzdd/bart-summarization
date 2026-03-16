from transformers import BartTokenizer
from config import *
from datasets import load_dataset

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

def load_and_filter_data():
    data = load_dataset(DATASET_NAME, DATASET_VERSION)

    for split in ['train','test','validation']:
        data[split] = data[split].filter(lambda x : MIN_SUMMARY_LEN < len(x['highlights'].split()) < MAX_SUMMARY_LEN and
                                MIN_ARTICLE_LEN < len(x['article'].split()) < MAX_ARTICLE_LEN)
        
    training_data = data['train'].select(range(TRAIN_SIZE))
    validation_data = data['validation'].select(range(VAL_SIZE))

    return training_data, validation_data

def preprocess(example):
    model_input = tokenizer(
        example['article'],
        padding = 'max_length',
        truncation = True,
        max_length = MAX_INPUT_LENGTH
    )

    labels = tokenizer(
        example['highlights'],
        padding = 'max_length',
        truncation = True,
        max_length = MAX_TARGET_LENGTH
    )

    model_input['labels'] = labels['input_ids']

    return model_input

def fixlabels(example):
    example['labels'] = [ 
        -100 if token == tokenizer.pad_token_id else token
        for token in example['labels']
    ]
    return example

def get_datasets():
    train , val = load_and_filter_data()

    train_data = train.map(
        preprocess,
        batched = False,
        remove_columns= ['id', 'article', 'highlights']
    )
    val_data = val.map(
        preprocess,
        batched = False,
        remove_columns= ['id', 'article', 'highlights']
    )

    train_data = train_data.map(
        fixlabels,
        batched = False
    )
    val_data = val_data.map(
        fixlabels,
        batched = False
    )

    return train_data , val_data

