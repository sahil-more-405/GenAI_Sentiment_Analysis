from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import torch.nn.functional as F

# Load the fine-tuned model and tokenizer
model_path = "./filtered_data_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Define the label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict(text: str) -> dict:
    """
    This function takes a text string as input, tokenizes it,
    passes it through the DistilBERT model, and returns the predicted sentiment,
    confidence score, and raw probabilities.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs.logits, dim=1)

    # Get the predicted class index and confidence score
    confidence, predicted_class_idx = torch.max(probabilities, dim=1)

    predicted_class_idx = predicted_class_idx.item()
    confidence = confidence.item()

    # Map the index to the corresponding label
    sentiment = label_map[predicted_class_idx]

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "raw_confidence": probabilities.squeeze().tolist()
    }