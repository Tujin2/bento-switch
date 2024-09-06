import asyncio
import json
import logging
from typing import Any, List, Dict

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from service import app

# Mock Model Wrapper
class MockModelWrapper(BaseModel):
    def __init__(self, model_name: str, model_path: str, default_params: Dict = None):
        pass

    def initialize_model(self):
        pass

    def load_model(self):
        pass

    def create_prompt(self, messages: List[dict]) -> str:
        return "Mock Prompt"

    def get_response(self, prompt: str, **kwargs) -> Any:
        return "Mock Raw Response"

    def format_output(self, raw_output: Any) -> dict:
        return {"choices": [{"text": "Mock Formatted Response"}]}

    def cleanup(self):
        pass


@pytest.fixture
def client():
    app.state.model_manager.model_wrappers["mock_model"] = MockModelWrapper(
        model_name="mock_model", model_path="mock_path"
    )
    app.state.model_manager.current_model_name = "mock_model"
    with TestClient(app) as client:
        yield client


def test_list_models(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "object": "list",
        "data": [{"id": "mock_model", "object": "model", "created": 1677610602, "owned_by": "organization-owner"}],
    }


def test_service_info(client: TestClient):
    response = client.get("/service-info")
    assert response.status_code == 200
    assert response.text == "Service is using model: mock_model"


def test_create_chat_completion(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock_model",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
    )
    assert response.status_code == 200
    assert response.text.startswith("data: ")
    data = json.loads(response.text.replace("data: ", ""))
    assert data == {"choices": [{"text": "Mock Formatted Response"}]}