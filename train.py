import os, html, shutil
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.transformers
from mlflow import MlflowClient
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

# Configure mlflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("sentiment-analysis")

# ---------- settings ----------
MODEL_NAME = "distilbert-base-uncased"
MAX_ROWS = None
EPOCHS = 1
MAX_LEN = 64
LEARNING_RATE = 5e-5
MIN_TEST_ACCURACY = 0.70   # quality gate, set just below your real accuracy after the first run

# ---------- device ----------
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found, using CPU")
    torch.set_num_threads(os.cpu_count())
BATCH = 64 if USE_GPU else 32

# ---------- load ----------
train = pd.read_csv("train.csv", encoding="latin1")
test = pd.read_csv("test.csv", encoding="latin1")

train = train.dropna(subset=["text", "sentiment"])
test = test.dropna(subset=["text", "sentiment"])

# ---------- clean text and map labels ----------
label_map = {"negative": 0, "neutral": 1, "positive": 2}
labels = {v: k for k, v in label_map.items()}

for df in (train, test):
    df["text"] = (
        df["text"].astype(str).apply(html.unescape)
        .str.replace(r"http\S+|www\S+", "", regex=True)
        .str.replace(r"@\w+", "", regex=True)
        .str.replace(r"<.*?>", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["sentiment"] = df["sentiment"].str.lower().map(label_map)
    df.dropna(subset=["sentiment"], inplace=True)
    df["sentiment"] = df["sentiment"].astype(int)

if MAX_ROWS:
    train = train.sample(min(MAX_ROWS, len(train)), random_state=42)

# ---------- split train into train and validation ----------
X_train, X_val, y_train, y_val = train_test_split(
    train["text"].tolist(), train["sentiment"].tolist(),
    test_size=0.2, random_state=42, stratify=train["sentiment"])

# ---------- tokenize (padding is done per batch by the collator) ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenizer(X_train, truncation=True, max_length=MAX_LEN)
val_enc = tokenizer(X_val, truncation=True, max_length=MAX_LEN)
test_enc = tokenizer(test["text"].tolist(), truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_dict({**dict(train_enc), "labels": y_train})
val_ds = Dataset.from_dict({**dict(val_enc), "labels": y_val})
test_ds = Dataset.from_dict({**dict(test_enc), "labels": test["sentiment"].tolist()})

# ---------- model ----------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3,
    id2label=labels, label2id=label_map)

# Trainer requires this as a function
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, preds),
            "f1": f1_score(pred.label_ids, preds, average="weighted")}

args = TrainingArguments(
    output_dir="./results",          # stays empty, no checkpoints are saved
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH * 2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=USE_GPU,
    report_to="mlflow",              # logs loss and val metrics per epoch
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ---------- tracked training run ----------
with mlflow.start_run() as run:

    # record the training configuration
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "max_len": MAX_LEN,
        "batch_size": BATCH,
        "learning_rate": LEARNING_RATE,
        "max_rows": MAX_ROWS,
        "device": "gpu" if USE_GPU else "cpu",
    })

    # train
    trainer.train()

    # test on test.csv
    test_res = trainer.evaluate(test_ds, metric_key_prefix="test")
    accuracy = test_res["test_accuracy"]
    f1 = test_res["test_f1"]
    print(f"\nTest Accuracy: {accuracy*100:.2f}%   Test F1: {f1:.4f}")

    test_pred = trainer.predict(test_ds)
    test_preds = np.argmax(test_pred.predictions, axis=1)
    print(classification_report(test["sentiment"], test_preds,
                                target_names=["negative", "neutral", "positive"]))

    # Model Quality Gate
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected: test accuracy {accuracy:.2f} "
            f"is below the required {MIN_TEST_ACCURACY}"
        )
    print("Model Quality Gate passed ......")

    mlflow.log_metrics({"test_accuracy": accuracy, "test_f1": f1})

    # save the model locally
    model_path = "models/sentiment_bert"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Model saved in:", model_path)

    # model registration
    model_info = mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        name="sentiment-model",
        task="text-classification",
        registered_model_name="sentiment-classifier",
    )

    # Assign the champion alias to the newly registered model version
    client = MlflowClient()
    client.set_registered_model_alias(
        name="sentiment-classifier",
        alias="champion",
        version=model_info.registered_model_version,
    )
    print("Registered Model : sentiment-classifier")