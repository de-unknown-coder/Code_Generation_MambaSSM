from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_generate_success():
    response = client.post(
        "/generate",
        json={
            "instruction": "Write a factorial function",
            "input": "5"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "generated_code" in data
    assert isinstance(data["generated_code"], str)

def test_empty_instruction():
    response = client.post(
        "/generate",
        json={
            "instruction": "   ",
            "input": "5"
        }
    )

    assert response.status_code == 422

def test_long_instruction():
    response = client.post(
        "/generate",
        json={
            "instruction": "a" * 301,
            "input": "5"
        }
    )

    assert response.status_code == 422


def test_long_input():
    response = client.post(
        "/generate",
        json={
            "instruction": "Write factorial",
            "input": "a" * 501
        }
    )

    assert response.status_code == 422
    