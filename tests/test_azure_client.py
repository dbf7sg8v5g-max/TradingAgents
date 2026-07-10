"""Tests for AzureOpenAIClient — the only LLM provider client with no prior
dedicated test file.

Follows the ``create_llm_client(provider=..., model=..., **kwargs).get_llm()``
+ attribute-assertion pattern used by ``test_openai_compatible_provider.py``.
Construction requires ``AZURE_OPENAI_ENDPOINT``/``OPENAI_API_VERSION`` (Azure's
own SDK-level validation, not this codebase's), so every test sets them.
"""

import pytest

from tradingagents.llm_clients.azure_client import AzureOpenAIClient
from tradingagents.llm_clients.factory import create_llm_client

_REQUIRED_ENV = {
    "AZURE_OPENAI_API_KEY": "placeholder",
    "AZURE_OPENAI_ENDPOINT": "https://myresource.openai.azure.com/",
    "OPENAI_API_VERSION": "2025-03-01-preview",
}


@pytest.fixture(autouse=True)
def _azure_env(monkeypatch):
    for key, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.mark.unit
def test_factory_routes_to_azure_client():
    client = create_llm_client(provider="azure", model="gpt-4o")
    assert type(client).__name__ == "AzureOpenAIClient"


@pytest.mark.unit
def test_deployment_name_env_overrides_model(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deployment")
    llm = AzureOpenAIClient(model="gpt-4o").get_llm()
    assert llm.deployment_name == "my-deployment"


@pytest.mark.unit
def test_deployment_name_falls_back_to_model_when_unset(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
    llm = AzureOpenAIClient(model="gpt-4o").get_llm()
    assert llm.deployment_name == "gpt-4o"


@pytest.mark.unit
def test_endpoint_and_api_version_are_sdk_env_driven():
    llm = AzureOpenAIClient(model="gpt-4o").get_llm()
    assert str(llm.azure_endpoint) == "https://myresource.openai.azure.com/"
    assert llm.openai_api_version == "2025-03-01-preview"


@pytest.mark.unit
def test_passthrough_kwargs_land_on_instance():
    llm = AzureOpenAIClient(
        model="gpt-4o", temperature=0.3, timeout=10, max_retries=2
    ).get_llm()
    assert llm.temperature == 0.3
    assert llm.request_timeout == 10
    assert llm.max_retries == 2


@pytest.mark.unit
def test_validate_model_accepts_any_model_name():
    client = AzureOpenAIClient(model="literally-anything")
    assert client.validate_model() is True


@pytest.mark.unit
def test_warn_if_unknown_model_does_not_raise():
    client = AzureOpenAIClient(model="some-custom-deployment")
    client.warn_if_unknown_model()  # must not raise


@pytest.mark.unit
def test_returned_instance_is_normalized_azure_chat_openai():
    llm = AzureOpenAIClient(model="gpt-4o").get_llm()
    assert type(llm).__name__ == "NormalizedAzureChatOpenAI"
