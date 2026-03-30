import os
from argparse import Namespace

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, MODEL_NAME
from inference import load_model_and_tokenizer, summarize


class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str



app = FastAPI(title="BART Summarization API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", MODEL_NAME)
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))
MIN_SUMMARY_LENGTH = int(os.getenv("MIN_SUMMARY_LENGTH", "20"))
MAX_INPUT_LENGTH_VALUE = int(os.getenv("MAX_INPUT_LENGTH", str(MAX_INPUT_LENGTH)))
MAX_SUMMARY_LENGTH_VALUE = int(os.getenv("MAX_SUMMARY_LENGTH", str(MAX_TARGET_LENGTH)))

model = None
tokenizer = None
device = None

@app.on_event('startup')
def load_resources():
    global model , tokenizer , device
    model , tokenizer , device = load_model_and_tokenizer(MODEL_PATH)

@app.get('/health')
def health_check():
    return {'status':'ok'}

@app.post('/summarize', response_model=SummarizeResponse)
def summarize_text(payload: SummarizeRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail = 'Text must not be empty.')
    
    generation_args = Namespace(
        num_beams = NUM_BEAMS,
        max_input_length = MAX_INPUT_LENGTH_VALUE,
        max_summary_length = MAX_SUMMARY_LENGTH_VALUE,
        min_summary_length = MIN_SUMMARY_LENGTH
    )

    summary = summarize(payload.text, model, tokenizer, device, generation_args)
    return SummarizeResponse(summary=summary)

