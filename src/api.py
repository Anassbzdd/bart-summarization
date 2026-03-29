import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer

from config import MODEL_NAME


class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str


class ModelService:
    def __init__(self, model_path=MODEL_NAME):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load(self):
        self.model = BartForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_path)

    def summarize(
        self,
        text,
        max_input_length=1024,
        max_summary_length=128,
        min_summary_length=20,
        num_beams=4,
    ):
        model_inputs = self.tokenizer(
            text,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        output = self.model.generate(
            model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


service = ModelService(model_path=os.getenv("MODEL_PATH", MODEL_NAME))


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    app.state.service = service
    yield


app = FastAPI(title="BART Summarization API", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    summary = service.summarize(text)
    return SummarizeResponse(summary=summary)
