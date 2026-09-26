import argparse
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_FILE_URL = "https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin"
MODEL_FILE = Path("pytorch_model.bin")


def download_model():
    #Download the model file using requests if it is not already present.
    if MODEL_FILE.exists():
        print("Model file already exists.")
        return

    print("Downloading model file...")
    response = requests.get(MODEL_FILE_URL, stream=True, timeout=60)
    response.raise_for_status()

    with MODEL_FILE.open("wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    print("Model downloaded successfully.")


def predict_sentiment(sentence):
    #Predict positive or negative sentiment for a sentence.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return model.config.id2label[predicted_class]


def main():
    parser = argparse.ArgumentParser(
        description="Simple DistilBERT sentiment classification app."
    )
    parser.add_argument(
        "sentence",
        nargs="*",
        help="Sentence to classify. If omitted, example sentences are used.",
    )
    args = parser.parse_args()

    download_model()

    if args.sentence:
        sentences = [" ".join(args.sentence)]
    else:
        sentences = [
            "I love this product!",
            "This movie was terrible.",
        ]

    for sentence in sentences:
        sentiment = predict_sentiment(sentence)
        print(f"Sentence: {sentence}")
        print(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()
