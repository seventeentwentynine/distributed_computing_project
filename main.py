# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = FastAPI(title="Fact-Checking RoBERTa API")


# Define request schema
class PredictRequest(BaseModel):
    text: str


# Global variables
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated label mapping based on your training script groupings
id_to_labels = {
    0: "False / Mostly False / Pants on Fire",
    1: "True / Mostly True",
    2: "Half True"
}


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model_dir = "./merged_model"

    print(f"Loading custom tokenizer and model to {device}...")
    # Using AutoTokenizer ensures it reads your special tokens correctly
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()
    print("Model and Tokenizer loaded successfully.")


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Tokenize input
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    print(inputs)
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class_id = int(np.argmax(probabilities))

    predicted_label = id_to_labels.get(predicted_class_id, "Unknown")

    return {
        "predicted_label": predicted_label,
        "confidence": float(probabilities[predicted_class_id]),
        "probabilities": {
            id_to_labels[i]: float(prob) for i, prob in enumerate(probabilities)
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}