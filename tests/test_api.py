# Import testclient so we can send simulated http requests
from fastapi.testclient import TestClient

import pytest

# import the application factory defined in main.py
from main import create_app

# These tests do not need a running MLflow server or the real model.
# They test the API on its own, using a fake model.

received = []   # remembers the text the fake model was given


def make_fake_model(label="positive", score=0.987654):
    def fake_model(text, **kwargs):
        received.append(text)
        return [{"label": label, "score": score}]
    return fake_model


def fake_version_loader():
    return "test-version"


@pytest.fixture
def client():
    received.clear()
    app = create_app(
        model_loader=make_fake_model,
        version_loader=fake_version_loader,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """verify that the application is running and reports the model details"""
    response = client.get("/health")
    assert response.status_code == 200

    assert response.json() == {
        "status":"healthy",
        "model_name": "sentiment-classifier",
        "model_version":"test-version",
        
        }


def test_valid_prediction(client):
    """verify that valid text produces a prediction"""
    response = client.post("/predict", json={"text": "I love this phone, it's amazing!"})
    assert response.status_code == 200

    result = response.json()

    # check the response contract: both fields must exist
    assert "class_name" in result
    assert "confidence" in result

    # the class must be one of the three sentiment classes
    assert result["class_name"] in ["negative", "neutral", "positive"]

    # confidence must be a probability
    assert 0 <= result["confidence"] <= 1


def test_text_is_cleaned_before_prediction(client):
    """verify that urls and @mentions are removed, like in training"""
    client.post("/predict", json={"text": "Great phone!!  http://x.com  @bob"})
    assert received[-1] == "Great phone!!"



def test_text_length_validation(client):
    """verify that empty text and text over 1000 characters are rejected"""
    for text in ["", "a" * 1001]:
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 422