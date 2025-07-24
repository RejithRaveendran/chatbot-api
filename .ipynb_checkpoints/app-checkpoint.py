
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os

app = FastAPI()

# Step 1: Define local model directory
model_dir = "trained_chatbot_model"  # Ensure this folder exists locally with all necessary files

# Step 2: Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Step 3: Load label map
with open(f"{model_dir}/label_map.json", "r") as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# Step 4: Define input schema
class Query(BaseModel):
    question: str

# Step 5: Define prediction endpoint
@app.post("/predict")
def predict(query: Query):
    inputs = tokenizer(query.question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    answer = reverse_label_map[predicted_label]
    return {"answer": answer}
