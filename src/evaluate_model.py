import argparse
from config import *

def argparse():
    parser = argparse.Argumentparser(description = 'Evaluate a summarization model with ROUGE.')
    parser.add_argument('--model-path', default= MODEL_NAME)
    parser.add_argument('--test-size',type=int, default= TEST_SIZE)
    parser.add_argument('--max-input-length',type=int, default= MAX_INPUT_LENGTH)
    parser.add_argument('--max-summary-length',type=int, default= MAX_SUMMARY_LEN)
    parser.add_argument('--min-summary-length',type=int, default= MIN_SUMMARY_LEN)
    parser.add_argument('--save-path',default= None)
    parser.add_argument('--num-beams',type=int,default= 4)
    return parser.argparse()

def summarize(model, article , tokenizer, device)

