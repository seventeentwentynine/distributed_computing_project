# main.py
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizerFast
import time
import tritonclient.grpc as grpcclient

app = FastAPI(title="Fact-Checking RoBERTa API (Triton Frontend)")

# --- Configuration ---
# Read the Triton URL from the environment, default to localhost for local testing
TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")
MODEL_NAME = "roberta_news"
TOKENIZER_DIR = "./merged_model"

# Label mapping from your training script
id_to_labels = {
    0: "False / Mostly False",
    1: "True / Mostly True",
    2: "Unsure / Half True"
}

# Global variables
tokenizer = None
triton_client = None


class PredictRequest(BaseModel):
    text: str


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@app.on_event("startup")
def startup_event():
    global tokenizer, triton_client

    print(f"Loading custom tokenizer from {TOKENIZER_DIR}...")
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR)

    print(f"Connecting to Triton Inference Server at {TRITON_URL}...")
    try:
        # Establish a persistent gRPC connection to Triton
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
        if not triton_client.is_server_ready():
            print("WARNING: Triton server is not ready yet.")
    except Exception as e:
        print(f"Failed to connect to Triton: {e}")


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if not triton_client:
        raise HTTPException(status_code=500, detail="Triton client not initialized.")

    # 1. Tokenize the input text (Runs on CPU)
    # return_tensors="np" gives us standard NumPy arrays instead of PyTorch tensors
    encoded = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    # Triton requires strictly typed INT64 arrays based on our config.pbtxt
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    # 2. Package the data into Triton's gRPC Protobuf format
    inputs = [
        grpcclient.InferInput('input_ids', input_ids.shape, "INT64"),
        grpcclient.InferInput('attention_mask', attention_mask.shape, "INT64")
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    start_time = time.time()
    # 3. Send the binary stream to Triton
    try:
        results = triton_client.infer(model_name=MODEL_NAME, inputs=inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Triton Inference Error: {str(e)}")

    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000

    # 4. Unpack the binary response
    logits = results.as_numpy('logits')[0]  # Extract the first item in the batch

    # 5. Calculate probabilities
    probabilities = softmax(logits)
    predicted_class_id = int(np.argmax(probabilities))
    predicted_label = id_to_labels.get(predicted_class_id, "Unknown")

    return {
        "predicted_label": predicted_label,
        "processing_time_ms": processing_time_ms,
        "confidence": float(probabilities[predicted_class_id]),
        "probabilities": {
            id_to_labels[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        "backend": "Triton gRPC"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}