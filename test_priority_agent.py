"""Unit tests for priority_agent.py using pytest.

These tests use mocked Claude responses to test priority scoring, payload formatting,
error handling, and retries on malformed JSON without requiring network access or
an active Anthropic API key.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from classifier_agent import classify_email
from priority_agent import (
    MODEL_NAME,
    PRIORITY_SYSTEM_PROMPT,
    score_priority,
)


def _create_mock_response(text: str) -> MagicMock:
    """Helper to generate a mock Anthropic API response object."""
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = text

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    return mock_resp


def test_score_priority_p1_critical_delay() -> None:
    """Test priority scoring for a high-urgency delay complaint with VIP customer tier."""
    classifier_output = {
        "category": "delay_complaint",
        "intent": "critical delay halting assembly line",
        "awb": "AWB-998877",
        "sentiment": "urgent",
        "language": "en",
        "confidence": 0.98,
    }
    customer_tier = "Platinum"

    expected_output = {
        "priority": "P1",
        "score": 88,
        "reply_deadline": "15 min",
        "reasons": [
            "Halted production line (penalty risk 15/15)",
            "Delayed critical components (22/25)",
            "Platinum VIP customer tier (15/15)",
            "Urgent sentiment (18/20)",
        ],
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(expected_output)
    )

    result = score_priority(
        classifier_output=classifier_output,
        customer_tier=customer_tier,
        client=mock_client,
    )

    assert result == expected_output
    assert result["priority"] == "P1"
    assert result["score"] >= 75
    assert result["reply_deadline"] == "15 min"
    assert isinstance(result["reasons"], list)

    # Check API call arguments
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == MODEL_NAME
    assert call_kwargs["system"] == PRIORITY_SYSTEM_PROMPT

    # Verify user message payload contains JSON with classifier_output and customer_tier
    sent_messages = call_kwargs["messages"]
    assert len(sent_messages) == 1
    sent_payload = json.loads(sent_messages[0]["content"])
    assert sent_payload["customer_tier"] == "Platinum"
    assert sent_payload["classifier_output"] == classifier_output


def test_score_priority_p4_low_urgency() -> None:
    """Test priority scoring for low-urgency category 'other'."""
    classifier_output = {
        "category": "other",
        "intent": "casual greeting or general inquiry",
        "awb": None,
        "sentiment": "neutral",
        "language": "en",
        "confidence": 0.3,
    }
    customer_tier = "Standard"

    expected_output = {
        "priority": "P4",
        "score": 15,
        "reply_deadline": "24hr",
        "reasons": [
            "General inquiry unrelated to active shipping (0/25)",
            "Standard tier customer (3/15)",
            "No financial or operational risk (5/15)",
            "Neutral sentiment (2/20)",
        ],
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(expected_output)
    )

    result = score_priority(
        classifier_output=classifier_output,
        customer_tier=customer_tier,
        client=mock_client,
    )

    assert result == expected_output
    assert result["priority"] == "P4"
    assert result["score"] < 25
    assert result["reply_deadline"] == "24hr"


def test_score_priority_malformed_json_retry_success() -> None:
    """Test retry mechanism when priority agent receives malformed JSON on first attempt."""
    classifier_output = {
        "category": "tracking_query",
        "intent": "check customs clearance status",
        "awb": "AWB-443322",
        "sentiment": "neutral",
        "language": "en",
        "confidence": 0.95,
    }
    customer_tier = "Gold"

    malformed_output = "Output: {priority: P2, score: 60, reply_deadline: 1hr"
    valid_output = {
        "priority": "P2",
        "score": 60,
        "reply_deadline": "1hr",
        "reasons": ["Gold tier customer", "Customs clearance query in transit"],
    }

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _create_mock_response(malformed_output),
        _create_mock_response(json.dumps(valid_output)),
    ]

    result = score_priority(
        classifier_output=classifier_output,
        customer_tier=customer_tier,
        client=mock_client,
        max_retries=1,
    )

    assert result == valid_output
    assert mock_client.messages.create.call_count == 2

    # Verify conversation history in retry request
    retry_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    messages = retry_call_kwargs["messages"]
    assert len(messages) == 3
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert "not valid JSON" in messages[2]["content"]


def test_score_priority_max_retries_exhausted_raises_error() -> None:
    """Test that ValueError is raised if JSON is persistently malformed."""
    classifier_output = {"category": "other"}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response("Invalid JSON stream")

    with pytest.raises(ValueError, match="Failed to obtain valid JSON"):
        score_priority(
            classifier_output=classifier_output,
            customer_tier="Bronze",
            client=mock_client,
            max_retries=2,
        )

    assert mock_client.messages.create.call_count == 3


def test_score_priority_markdown_code_block_parsing() -> None:
    """Test that JSON wrapped in markdown fences parses without requiring retries."""
    classifier_output = {
        "category": "booking_or_quote_request",
        "intent": "rate inquiry for 2 pallets",
        "awb": None,
        "sentiment": "neutral",
        "language": "en",
        "confidence": 0.9,
    }
    expected_output = {
        "priority": "P3",
        "score": 35,
        "reply_deadline": "4hr",
        "reasons": ["New booking quote inquiry", "Standard customer response window"],
    }

    wrapped_content = f"```json\n{json.dumps(expected_output)}\n```"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(wrapped_content)

    result = score_priority(
        classifier_output=classifier_output,
        customer_tier="Standard",
        client=mock_client,
    )

    assert result == expected_output
    assert mock_client.messages.create.call_count == 1


def test_classifier_to_priority_pipeline_integration() -> None:
    """Test chaining output from classify_email directly into score_priority."""
    raw_email = "AWB-100200 is delayed 3 days! Need it for factory ASAP!"

    classified = {
        "category": "delay_complaint",
        "intent": "urgent delay resolution",
        "awb": "AWB-100200",
        "sentiment": "urgent",
        "language": "en",
        "confidence": 0.97,
    }

    prioritized = {
        "priority": "P1",
        "score": 85,
        "reply_deadline": "15 min",
        "reasons": ["Factory delay", "Urgent sentiment"],
    }

    classifier_client = MagicMock()
    classifier_client.messages.create.return_value = _create_mock_response(
        json.dumps(classified)
    )

    priority_client = MagicMock()
    priority_client.messages.create.return_value = _create_mock_response(
        json.dumps(prioritized)
    )

    # Step 1: Classify
    class_res = classify_email(raw_email, client=classifier_client)
    assert class_res["category"] == "delay_complaint"

    # Step 2: Score Priority
    prio_res = score_priority(
        classifier_output=class_res,
        customer_tier="Platinum",
        client=priority_client,
    )
    assert prio_res["priority"] == "P1"
    assert prio_res["score"] == 85
