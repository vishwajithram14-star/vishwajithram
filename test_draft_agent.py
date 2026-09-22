"""Unit tests for draft_agent.py using pytest.

Tests verify:
- That refund/discount dollar amounts or percentages are never promised in draft_reply text.
- Proposed actions are strictly limited to the approved action whitelist.
- Claude model name ("claude-sonnet-4-6") and exact system prompt are used.
- JSON parsing, markdown fences, and retry handling.
- Full 4-agent pipeline integration test (classifier -> priority -> rag -> draft).
"""

from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from classifier_agent import classify_email
from draft_agent import (
    ALLOWED_ACTIONS,
    DRAFT_SYSTEM_PROMPT,
    MODEL_NAME,
    REFUND_AMOUNT_PATTERN,
    draft_reply,
)
from priority_agent import score_priority
from rag_agent import research_facts


def _create_mock_response(text: str) -> MagicMock:
    """Helper to create a mock Claude response object."""
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = text

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    return mock_resp


def test_draft_reply_no_refund_amount_promised() -> None:
    """Test that draft_reply never promises an explicit refund amount or discount percentage."""
    email_text = "Shipment AWB-8849201 is 3 days late. Give me my 500 dollars refund immediately!"
    facts = [
        {
            "fact": "AWB-8849201 is delayed at Kansas City Hub due to rail mechanical delay.",
            "source": "live_tracking_tool",
        },
        {
            "fact": "Express shipments delayed over 48 hours are eligible for a 20% freight credit review.",
            "source": "delay_policy.txt",
        },
        {
            "fact": "Delay complaints must be formally submitted within 14 business days.",
            "source": "delay_policy.txt",
        },
    ]

    expected_output = {
        "draft_reply": (
            "Dear Customer, thank you for reaching out regarding AWB-8849201. "
            "We apologize for the delay caused by a mechanical hold at Kansas City Hub. "
            "Regarding your compensation request, delayed shipments are eligible for credit review. "
            "We have submitted a request for refund review with our billing department on your behalf."
        ),
        "proposed_actions": ["request_refund_review", "reschedule_delivery"],
        "facts_used": [
            "AWB-8849201 delayed at Kansas City Hub",
            "Eligible for credit review within 14 business days",
        ],
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(expected_output)
    )

    result = draft_reply(email_text=email_text, facts=facts, client=mock_client)

    # Verification: Ensure no dollar amount or refund percentage is promised in the draft reply
    reply_text = result["draft_reply"]
    assert "$" not in reply_text
    assert not REFUND_AMOUNT_PATTERN.search(reply_text), (
        f"Found forbidden refund amount in draft reply: {reply_text}"
    )

    # Check proposed action is request_refund_review rather than a direct refund promise
    assert "request_refund_review" in result["proposed_actions"]


def test_draft_reply_allowed_actions_whitelist() -> None:
    """Test that all proposed actions in output belong strictly to the allowed action whitelist."""
    email_text = "Where is my package AWB-772199?"
    facts = [
        {
            "fact": "AWB-772199 is Out for Delivery in Chicago North with delivery expected today by 17:00 CST.",
            "source": "live_tracking_tool",
        }
    ]

    expected_output = {
        "draft_reply": "Hello, AWB-772199 is out for delivery and scheduled to arrive by 17:00 CST today.",
        "proposed_actions": ["reply_only"],
        "facts_used": ["AWB-772199 is Out for Delivery by 17:00 CST"],
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(expected_output)
    )

    result = draft_reply(email_text=email_text, facts=facts, client=mock_client)

    for action in result["proposed_actions"]:
        assert action in ALLOWED_ACTIONS, f"Unauthorized action proposed: {action}"


def test_draft_reply_disallowed_actions_sanitized() -> None:
    """Test that if the model proposes an invalid action, it is sanitized against ALLOWED_ACTIONS."""
    email_text = "Cargo damaged on delivery."
    facts = [{"fact": "Cargo damaged", "source": "damage_claims.txt"}]

    raw_from_model = {
        "draft_reply": "We apologize for the damaged package.",
        "proposed_actions": ["raise_damage_claim", "unauthorized_custom_action", "create_ticket"],
        "facts_used": ["Cargo damaged"],
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(raw_from_model)
    )

    result = draft_reply(email_text=email_text, facts=facts, client=mock_client)

    assert "unauthorized_custom_action" not in result["proposed_actions"]
    assert "raise_damage_claim" in result["proposed_actions"]
    assert "create_ticket" in result["proposed_actions"]


def test_draft_reply_parameters_and_system_prompt() -> None:
    """Test that Claude API is invoked with correct model name and exact system prompt."""
    email_text = "Can I change my delivery address for AWB-551-09827461?"
    facts = [
        {
            "fact": "Address reroute fee is $15.00 and must be requested before Out for Delivery.",
            "source": "faq.txt",
        }
    ]

    expected_output = {
        "draft_reply": (
            "You may modify your delivery address prior to 'Out for Delivery' status. "
            "Please note a $15.00 administrative reroute fee applies."
        ),
        "proposed_actions": ["create_ticket"],
        "facts_used": ["Address reroute fee is $15.00 prior to Out for Delivery"],
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(expected_output)
    )

    draft_reply(email_text=email_text, facts=facts, client=mock_client)

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == MODEL_NAME
    assert call_kwargs["system"] == DRAFT_SYSTEM_PROMPT

    user_content = call_kwargs["messages"][0]["content"]
    assert email_text in user_content
    assert "VERIFIED FACTS AVAILABLE" in user_content


def test_draft_reply_malformed_json_retry_success() -> None:
    """Test retry recovery when model produces malformed JSON on first attempt."""
    email_text = "Status update please."
    facts = [{"fact": "In transit", "source": "live_tracking_tool"}]

    malformed_json = "Here is the response: {draft_reply: Hello, proposed_actions: ['reply_only'"
    valid_output = {
        "draft_reply": "Hello, your package is currently in transit.",
        "proposed_actions": ["reply_only"],
        "facts_used": ["In transit"],
    }

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _create_mock_response(malformed_json),
        _create_mock_response(json.dumps(valid_output)),
    ]

    result = draft_reply(
        email_text=email_text,
        facts=facts,
        client=mock_client,
        max_retries=1,
    )

    assert result == valid_output
    assert mock_client.messages.create.call_count == 2


def test_draft_reply_max_retries_exhausted_raises_error() -> None:
    """Test that persistent malformed JSON raises ValueError."""
    email_text = "Status update please."
    facts = [{"fact": "In transit", "source": "live_tracking_tool"}]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response("Not JSON at all")

    with pytest.raises(ValueError, match="Failed to obtain valid JSON"):
        draft_reply(
            email_text=email_text,
            facts=facts,
            client=mock_client,
            max_retries=1,
        )

    assert mock_client.messages.create.call_count == 2


def test_full_four_agent_pipeline_integration() -> None:
    """Integration test verifying end-to-end flow across Classifier, Priority, RAG, and Draft agents."""
    raw_email = (
        "Subject: Urgent: Shipment AWB-8849201 missing!\n\n"
        "Our factory line is stopped. Where is our shipment AWB-8849201? "
        "It was due 3 days ago. We demand an immediate explanation!"
    )
    customer_tier = "Platinum"

    # Step 1: Classifier Agent Mock
    mock_classifier_output = {
        "category": "delay_complaint",
        "intent": "inquire about overdue shipment and demand status",
        "awb": "AWB-8849201",
        "sentiment": "angry",
        "language": "en",
        "confidence": 0.98,
    }
    classifier_client = MagicMock()
    classifier_client.messages.create.return_value = _create_mock_response(
        json.dumps(mock_classifier_output)
    )

    classification = classify_email(raw_email, client=classifier_client)
    assert classification["category"] == "delay_complaint"
    assert classification["awb"] == "AWB-8849201"

    # Step 2: Priority Agent Mock
    mock_priority_output = {
        "priority": "P1",
        "score": 90,
        "reply_deadline": "15 min",
        "reasons": ["Halted factory line", "Platinum tier customer", "Angry sentiment"],
    }
    priority_client = MagicMock()
    priority_client.messages.create.return_value = _create_mock_response(
        json.dumps(mock_priority_output)
    )

    priority_assessment = score_priority(
        classifier_output=classification,
        customer_tier=customer_tier,
        client=priority_client,
    )
    assert priority_assessment["priority"] == "P1"
    assert priority_assessment["reply_deadline"] == "15 min"

    # Step 3: RAG Agent Mock
    mock_rag_output = {
        "facts": [
            {
                "fact": "AWB-8849201 is held at Kansas City Hub due to rail mechanical delay, rescheduled for road transfer.",
                "source": "live_tracking_tool",
            },
            {
                "fact": "Shipments delayed over 48 hours are eligible for credit review.",
                "source": "delay_policy.txt",
            },
        ],
        "enough_facts_found": True,
    }
    rag_client = MagicMock()
    rag_client.messages.create.return_value = _create_mock_response(
        json.dumps(mock_rag_output)
    )
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Policy text"]],
        "metadatas": [[{"source": "delay_policy.txt"}]],
    }

    research = research_facts(
        category=classification["category"],
        email_text=raw_email,
        awb=classification["awb"],
        collection=mock_collection,
        client=rag_client,
    )
    assert research["enough_facts_found"] is True
    assert len(research["facts"]) == 2

    # Step 4: Draft Agent Mock
    mock_draft_output = {
        "draft_reply": (
            "Dear Customer,\n\n"
            "We apologize for the delay on AWB-8849201. Your cargo is currently at the Kansas City Hub "
            "undergoing road transfer following a rail mechanical delay.\n\n"
            "We have opened a ticket to monitor its transfer and requested a refund review for the delay credit.\n\n"
            "Best regards,\nLogistics Team"
        ),
        "proposed_actions": ["reschedule_delivery", "request_refund_review", "create_ticket"],
        "facts_used": [
            "AWB-8849201 held at Kansas City Hub",
            "Delayed shipments eligible for credit review",
        ],
    }
    draft_client = MagicMock()
    draft_client.messages.create.return_value = _create_mock_response(
        json.dumps(mock_draft_output)
    )

    draft_result = draft_reply(
        email_text=raw_email,
        facts=research["facts"],
        client=draft_client,
    )

    assert "draft_reply" in draft_result
    assert "$" not in draft_result["draft_reply"]
    assert "reschedule_delivery" in draft_result["proposed_actions"]
    assert "request_refund_review" in draft_result["proposed_actions"]
