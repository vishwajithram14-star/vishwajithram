"""Unit tests for rag_agent.py using pytest.

Tests cover:
- Mock tracking tool (found and not-found AWBs)
- ChromaDB vector retrieval with mocked query results
- Claude RAG fact extraction with live tool data and static chunks
- Malformed JSON recovery and retries
- Live ChromaDB collection setup and indexing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_agent import (
    MODEL_NAME,
    RAG_SYSTEM_PROMPT,
    get_tracking_status,
    research_facts,
    setup_chroma_db,
)


def _create_mock_response(text: str) -> MagicMock:
    """Helper to create a mock Claude response object."""
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = text

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    return mock_resp


def test_get_tracking_status_found() -> None:
    """Test tracking lookup for valid known AWBs."""
    status_1 = get_tracking_status("AWB-8849201")
    assert status_1["status"] == "Delayed - Rail Freight Hold"
    assert "Kansas City" in status_1["current_location"]

    # Test normalization without dashes
    status_normalized = get_tracking_status("AWB8849201")
    assert status_normalized["status"] == "Delayed - Rail Freight Hold"


def test_get_tracking_status_not_found() -> None:
    """Test tracking lookup for an unknown AWB."""
    status = get_tracking_status("UNKNOWN-999999")
    assert status["status"] == "Not Found"
    assert "No active tracking records found" in status["details"]


def test_research_facts_with_mocked_chromadb_query() -> None:
    """Test research_facts using a mocked ChromaDB query and live tracking tool data."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [
            [
                "If an express shipment is delayed by more than 48 hours due to carrier error, the shipper is entitled to a 20% freight credit.",
                "Delay complaints must be formally submitted within 14 business days.",
                "Under Force Majeure circumstances, financial compensation is waived.",
            ]
        ],
        "metadatas": [
            [
                {"source": "delay_policy.txt"},
                {"source": "delay_policy.txt"},
                {"source": "delay_policy.txt"},
            ]
        ],
    }

    mock_claude_output = {
        "facts": [
            {
                "fact": "AWB-8849201 is delayed at Kansas City Hub due to rail hold and rescheduled for road transfer.",
                "source": "live_tracking_tool",
            },
            {
                "fact": "Shipments delayed over 48 hours qualify for a 20% freight credit.",
                "source": "delay_policy.txt",
            },
        ],
        "enough_facts_found": True,
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(mock_claude_output)
    )

    result = research_facts(
        category="delay_complaint",
        email_text="AWB-8849201 is late by 3 days. What compensation am I owed?",
        awb="AWB-8849201",
        collection=mock_collection,
        client=mock_client,
    )

    assert result == mock_claude_output
    assert result["enough_facts_found"] is True
    assert len(result["facts"]) == 2

    # Verify ChromaDB was queried for top 3 results
    mock_collection.query.assert_called_once()
    query_kwargs = mock_collection.query.call_args.kwargs
    assert query_kwargs["n_results"] == 3
    assert "delay_complaint" in query_kwargs["query_texts"][0]

    # Verify Claude parameters
    mock_client.messages.create.assert_called_once()
    claude_kwargs = mock_client.messages.create.call_args.kwargs
    assert claude_kwargs["model"] == MODEL_NAME
    assert claude_kwargs["system"] == RAG_SYSTEM_PROMPT

    # Verify user message contained both live tracking data and static documents
    user_content = claude_kwargs["messages"][0]["content"]
    assert "live_tracking_tool" in user_content
    assert "Kansas City Hub" in user_content
    assert "delay_policy.txt" in user_content


def test_research_facts_without_awb() -> None:
    """Test research_facts when no AWB is present in the customer email."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [
            [
                "Formal damage claims must be filed within 7 calendar days of package receipt.",
                "Original vendor commercial invoice verifying goods value is required.",
                "Standard carrier liability coverage is capped at $100.00 USD.",
            ]
        ],
        "metadatas": [
            [
                {"source": "damage_claims.txt"},
                {"source": "damage_claims.txt"},
                {"source": "damage_claims.txt"},
            ]
        ],
    }

    mock_claude_output = {
        "facts": [
            {
                "fact": "Damage claims must be filed within 7 calendar days of receipt.",
                "source": "damage_claims.txt",
            },
            {
                "fact": "Photographic evidence and original vendor commercial invoice are required.",
                "source": "damage_claims.txt",
            },
        ],
        "enough_facts_found": True,
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response(
        json.dumps(mock_claude_output)
    )

    result = research_facts(
        category="damage_or_loss_claim",
        email_text="Package arrived damaged. What documents do I need to file a claim?",
        awb=None,
        collection=mock_collection,
        client=mock_client,
    )

    assert result == mock_claude_output
    assert result["enough_facts_found"] is True

    # Check that live tracking was noted as not provided
    claude_kwargs = mock_client.messages.create.call_args.kwargs
    user_content = claude_kwargs["messages"][0]["content"]
    assert "No AWB provided" in user_content


def test_research_facts_malformed_json_retry_success() -> None:
    """Test retry recovery when model produces malformed JSON initially."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Routine customs clearance takes 24-48 hours."]],
        "metadatas": [[{"source": "faq.txt"}]],
    }

    malformed_reply = "Here is the response: {'facts': [{'fact': 'customs takes 24-48h'}"
    valid_reply = {
        "facts": [
            {"fact": "Routine customs clearance takes 24 to 48 hours.", "source": "faq.txt"}
        ],
        "enough_facts_found": True,
    }

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _create_mock_response(malformed_reply),
        _create_mock_response(json.dumps(valid_reply)),
    ]

    result = research_facts(
        category="documents_or_customs",
        email_text="How long will customs clearance take?",
        awb=None,
        collection=mock_collection,
        client=mock_client,
        max_retries=1,
    )

    assert result == valid_reply
    assert mock_client.messages.create.call_count == 2


def test_research_facts_max_retries_exhausted_raises_error() -> None:
    """Test that persistent malformed JSON raises ValueError."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Policy text"]],
        "metadatas": [[{"source": "faq.txt"}]],
    }

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _create_mock_response("Invalid text response")

    with pytest.raises(ValueError, match="Failed to obtain valid JSON"):
        research_facts(
            category="other",
            email_text="Hello",
            awb=None,
            collection=mock_collection,
            client=mock_client,
            max_retries=1,
        )

    assert mock_client.messages.create.call_count == 2


def test_setup_chroma_db_live() -> None:
    """Live test verifying ChromaDB loads policy documents and creates index."""
    collection = setup_chroma_db()
    assert collection is not None
    assert collection.count() > 0

    # Query the collection live
    res = collection.query(
        query_texts=["delay compensation SLA"],
        n_results=2,
    )
    assert len(res["documents"][0]) == 2
    assert any("delay" in doc.lower() for doc in res["documents"][0])
