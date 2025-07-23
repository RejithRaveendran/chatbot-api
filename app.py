
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

app = FastAPI()

# Load model and tokenizer
model_path = "trained_chatbot_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load label map
with open(f"{model_path}/label_map.json", "r") as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

class Query(BaseModel):
    question: str

@app.post("/predict")
def predict(query: Query):
    inputs = tokenizer(query.question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    answer = reverse_label_map[predicted_label]
    return {"answer": answer}
