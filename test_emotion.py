
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

print("Loading emotion classifier...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Get the correct emotion labels from the model itself.
EMOTIONS = model.config.id2label

sentences = [
    "I had a terrible day and feel lonely.",
    "I'm excited and want an adventurous movie.",
    "I'm in love and want a romantic movie.",
    "I feel peaceful and want something relaxing.",
]

for sentence in sentences:
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    # Find the three highest-scoring emotions.
    top_indices = torch.argsort(
        probabilities, descending=True
    )[:3]

    print(f"\nInput: {sentence}")

    for index in top_indices.tolist():
        label = EMOTIONS[index]
        score = probabilities[index].item()

        print(f"Emotion: {label}, Score: {score:.3f}")
