# Register the model manually by replacing all the files train.py generates into models/

import mlflow
import mlflow.transformers
from mlflow import MlflowClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("sentiment-analysis")

REGISTERED_MODEL_NAME = "sentiment-classifier"
MODEL_PATH = "models/sentiment_bert"

# paste the numbers Colab printed after training
'''
Update these values
'''
TEST_ACCURACY = 0.7948
TEST_F1 = 0.7953



model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

with mlflow.start_run(run_name="colab-model-import"):
    mlflow.set_tag("source", "trained on colab, imported manually")
    mlflow.log_params({
        "model_name": "distilbert-base-uncased",
        "max_len": 64,
        "device": "colab gpu",
    })
    mlflow.log_metrics({"test_accuracy": TEST_ACCURACY, "test_f1": TEST_F1})

    # log and register the model
    model_info = mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="sentiment-model",
        task="text-classification",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    # assign the champion alias
    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_info.registered_model_version,
    )
    print("Registered:", REGISTERED_MODEL_NAME, "version", model_info.registered_model_version)