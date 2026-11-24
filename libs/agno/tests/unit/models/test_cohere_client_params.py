"""Unit tests for Cohere client construction: client_params must reach the SDK clients."""

from unittest.mock import patch

import pytest

pytest.importorskip("cohere")

from agno.models.cohere.chat import Cohere


def test_client_params_forwarded_to_sync_client():
    model = Cohere(api_key="test-key", client_params={"timeout": 12.5, "base_url": "https://example.test"})

    with patch("agno.models.cohere.chat.CohereClient") as mock_client:
        model.get_client()

    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["timeout"] == 12.5
    assert kwargs["base_url"] == "https://example.test"


def test_client_params_forwarded_to_async_client():
    model = Cohere(api_key="test-key", client_params={"timeout": 12.5, "base_url": "https://example.test"})

    with patch("agno.models.cohere.chat.CohereAsyncClient") as mock_client:
        model.get_async_client()

    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["timeout"] == 12.5
    assert kwargs["base_url"] == "https://example.test"


def test_real_clients_accept_client_params():
    model = Cohere(api_key="test-key", client_params={"timeout": 12.5})

    assert model.get_client() is not None
    assert model.get_async_client() is not None
