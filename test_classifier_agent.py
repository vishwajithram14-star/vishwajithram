"""Unit tests for classifier_agent.py using pytest.

These tests use mocked responses to test classification, JSON parsing,
retries on malformed outputs, and parameter forwarding without requiring
network access or an active Anthropic API key.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from classifier_agent import (
    CLASSIFIER_SYSTEM_PROMPT,
    MODEL_NAME,
    _extract_and_parse_json,
    classify_email,
)


def _create_mock_response(text: str) -> MagicMock:
    """Helper to generate a mock Anthropic API response object."""
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = text

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    return mock_resp


def test_classify_email_delay_complaint() -> None:
    """Test classification for a clear delay complaint email."""
    email_text = (
        "Subject: Urgent shipment delayed AWB-772199\n\n"
        "Our delivery with AWB-772199 is 4 days late. Our factory is waiting "
        "and this is causing serious losses. What is going on?"
    )

    expected_output = {
        "category": "delay_complaint",
        "intent": "complain about delayed shipment and demand delivery status",
        "awb": "AWB-772199",
        "sentiment": "angry",
        "language": "en",
        "confidence": 0.98,
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(expected_output)
    )

    result = classify_email(email_text, client=mock_client)

    assert result == expected_output
    assert result["category"] == "delay_complaint"
    assert result["awb"] == "AWB-772199"
    assert result["sentiment"] in ["angry", "frustrated", "urgent"]
    assert result["confidence"] >= 0.9

    # Verify model and system prompt parameters
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == MODEL_NAME
    assert call_kwargs["system"] == CLASSIFIER_SYSTEM_PROMPT


def test_classify_email_low_confidence_other() -> None:
    """Test classification for an irrelevant email with low confidence category 'other'."""
    email_text = (
        "Subject: Quick question\n\n"
        "Hello, could you recommend any good coffee shops near downtown Chicago? "
        "Visiting next week. Thanks!"
    )

    expected_output = {
        "category": "other",
        "intent": "ask for non-logistics coffee shop recommendation",
        "awb": None,
        "sentiment": "neutral",
        "language": "en",
        "confidence": 0.2,
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(expected_output)
    )

    result = classify_email(email_text, client=mock_client)

    assert result == expected_output
    assert result["category"] == "other"
    assert result["awb"] is None
    assert result["confidence"] < 0.5


def test_classify_email_malformed_json_retry_success() -> None:
    """Test retry mechanism when model returns malformed JSON on first try and valid on second."""
    email_text = "Where is container AWB-12345? Is it stuck at customs?"

    malformed_response = "Here is your JSON response: {category: tracking_query, awb: AWB-12345"
    valid_output = {
        "category": "tracking_query",
        "intent": "track container location and check customs hold",
        "awb": "AWB-12345",
        "sentiment": "neutral",
        "language": "en",
        "confidence": 0.95,
    }

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _create_mock_response(malformed_response),
        _create_mock_response(json.dumps(valid_output)),
    ]

    result = classify_email(email_text, client=mock_client, max_retries=1)

    assert result == valid_output
    assert mock_client.messages.create.call_count == 2

    # Verify that the retry call includes conversation history correcting the format
    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    messages = second_call_kwargs["messages"]
    assert len(messages) == 3
    assert messages[0]["content"] == email_text
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert "not valid JSON" in messages[2]["content"]


def test_classify_email_max_retries_exhausted_raises_value_error() -> None:
    """Test that ValueError is raised if JSON remains invalid after max retries."""
    email_text = "Booking inquiry for 5 pallets to Frankfurt."

    invalid_text = "Sorry, I am unable to format as JSON right now."
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(invalid_text)

    with pytest.raises(ValueError, match="Failed to obtain valid JSON"):
        classify_email(email_text, client=mock_client, max_retries=2)

    # 1 initial call + 2 retries = 3 calls
    assert mock_client.messages.create.call_count == 3


def test_classify_email_markdown_code_block_parsing() -> None:
    """Test parsing when model wraps JSON inside markdown code blocks."""
    email_text = "Please send invoice copy for shipment AWB-998877."

    expected_output = {
        "category": "billing_or_invoice",
        "intent": "request invoice copy",
        "awb": "AWB-998877",
        "sentiment": "neutral",
        "language": "en",
        "confidence": 0.99,
    }

    wrapped_content = f"```json\n{json.dumps(expected_output)}\n```"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(wrapped_content)

    result = classify_email(email_text, client=mock_client)
    assert result == expected_output
    assert mock_client.messages.create.call_count == 1


def test_extract_and_parse_json_helper() -> None:
    """Test helper parsing function on various formats and edge cases."""
    # 1. Clean JSON
    clean_json = '{"category": "tracking_query", "confidence": 0.9}'
    assert _extract_and_parse_json(clean_json) == {
        "category": "tracking_query",
        "confidence": 0.9,
    }

    # 2. Markdown wrapped
    fenced_json = '```json\n{"category": "billing_or_invoice"}\n```'
    assert _extract_and_parse_json(fenced_json) == {"category": "billing_or_invoice"}

    # 3. JSON embedded in conversational commentary
    embedded_json = 'Sure! Here is the JSON:\n{"category": "documents_or_customs"}\nHope this helps!'
    assert _extract_and_parse_json(embedded_json) == {
        "category": "documents_or_customs"
    }

    # 4. Completely invalid text raises ValueError
    with pytest.raises(ValueError, match="Could not parse a valid JSON object"):
        _extract_and_parse_json("This is purely plain text with no braces.")
