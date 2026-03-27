from datasets import load_dataset
from config import *
from transformers import BartTokenizer

def tokenizer(model_name = MODEL_NAME):
    return BartTokenizer.from_pretrained(model_name)


def _filter_example(example):
    article_length = len(example['article'].split())
    summary_length = len(example['highlights'].split())
    return (
        MIN_ARTICLE_LEN < article_length < MAX_ARTICLE_LEN
        MIN_SUMMARY_LEN < summary_length < MAX_SUMMARY_LEN
    )

def load_dataset(train_size=None , val_size=None , test_size=None):
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION)

    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].filter(_filter_example)

    train_size = TRAIN_SIZE if train_size is None else train_size
    val_size = VAL_SIZE if val_size is None else val_size
    test_size = TEST_SIZE if test_size is None else test_size

    training_data = dataset.select(range(min(train_size, len(dataset['train']))))
    validation_data = dataset.select(range(min(val_size, len(dataset['validation']))))
    test_data = dataset.select(range(min(train_size, len(dataset['train']))))

    return training_data, validation_data, test_data

def preprocess_batch(example, tokenizer):
    input = tokenizer(
        example['article'],
        padding = 'max_length',
        max_length = MAX_ARTICLE_LEN,
        truncation = True
    )
    labels = tokenizer(
        example['article'],
        padding = 'max_length',
        max_length = MAX_TARGET_LENGTH,
        truncation = True
    )
     = 



