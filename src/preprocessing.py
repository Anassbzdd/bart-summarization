from datasets import load_dataset
from transformers import BartTokenizer

from config import (
    DATASET_NAME,
    DATASET_VERSION,
    MAX_ARTICLE_LEN,
    MAX_INPUT_LENGTH,
    MAX_SUMMARY_LEN,
    MAX_TARGET_LENGTH,
    MIN_ARTICLE_LEN,
    MIN_SUMMARY_LEN,
    MODEL_NAME,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)


def get_tokenizer(model_name=MODEL_NAME):
    return BartTokenizer.from_pretrained(model_name)


def _filter_example(example):
    article_length = len(example["article"].split())
    summary_length = len(example["highlights"].split())
    return (
        MIN_SUMMARY_LEN < summary_length < MAX_SUMMARY_LEN
        and MIN_ARTICLE_LEN < article_length < MAX_ARTICLE_LEN
    )


def load_and_filter_data(train_size=None, val_size=None, test_size=None):
    data = load_dataset(DATASET_NAME, DATASET_VERSION)

    for split in ["train", "validation", "test"]:
        data[split] = data[split].filter(_filter_example)

    train_size = TRAIN_SIZE if train_size is None else train_size
    val_size = VAL_SIZE if val_size is None else val_size
    test_size = TEST_SIZE if test_size is None else test_size

    training_data = data["train"].select(range(min(train_size, len(data["train"]))))
    validation_data = data["validation"].select(range(min(val_size, len(data["validation"]))))
    test_data = data["test"].select(range(min(test_size, len(data["test"]))))

    return training_data, validation_data, test_data


def preprocess_batch(examples, tokenizer):
    model_inputs = tokenizer(
        examples["article"],
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    labels = tokenizer(
        examples["highlights"],
        padding="max_length",
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    )

    model_inputs["labels"] = [
        [
            -100 if token == tokenizer.pad_token_id else token
            for token in label_ids
        ]
        for label_ids in labels["input_ids"]
    ]
    return model_inputs


def tokenize_split(dataset, tokenizer):
    return dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )


def get_datasets(tokenizer=None, train_size=None, val_size=None):
    tokenizer = tokenizer or get_tokenizer()
    train, val, _ = load_and_filter_data(train_size=train_size, val_size=val_size)
    return tokenize_split(train, tokenizer), tokenize_split(val, tokenizer)


def get_test_dataset(tokenizer=None, test_size=None):
    tokenizer = tokenizer or get_tokenizer()
    _, _, test = load_and_filter_data(test_size=test_size)
    return tokenize_split(test, tokenizer)
