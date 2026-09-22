"""RAG (Research) Agent for logistics email support system.

This module retrieves relevant company policy documents from ChromaDB using
sentence-transformers embeddings, fetches live tracking status from a mock tool
when an AWB is provided, and prompts Claude to extract verified facts needed to
respond to the customer.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import anthropic
import chromadb
from chromadb.utils import embedding_functions

from classifier_agent import MODEL_NAME, _extract_and_parse_json

# Exact system prompt for RAG Agent
RAG_SYSTEM_PROMPT = """You are the RAG (Research) Agent for a logistics company's email support system.
Given retrieved document chunks and/or live tool results, extract only the 
facts needed to answer the email. Return ONLY JSON:
{"facts": [{"fact": str, "source": str}], "enough_facts_found": bool}
Never invent a policy, date, or number not present in the sources.
Prefer live tool data over static documents when both exist for the same fact."""

# Hardcoded database of sample Air Waybills (AWBs)
SAMPLE_TRACKING_DATABASE: dict[str, dict[str, Any]] = {
    "AWB-8849201": {
        "awb": "AWB-8849201",
        "status": "Delayed - Rail Freight Hold",
        "current_location": "Kansas City Hub, MO",
        "destination": "Dallas, TX",
        "expected_delivery": "2026-09-24 14:00 CST",
        "last_update": "Mechanical delay on connecting rail freight line. Cargo rescheduled for priority road transfer.",
    },
    "551-09827461": {
        "awb": "551-09827461",
        "status": "Customs Clearance In Progress",
        "current_location": "Port of Long Beach Terminal 4, CA",
        "destination": "Los Angeles, CA",
        "expected_delivery": "2026-09-23 18:00 PST",
        "last_update": "Customs clearance documentation submitted to CBP. Standard 24-48hr review in progress.",
    },
    "AWB-772199": {
        "awb": "AWB-772199",
        "status": "Out for Delivery",
        "current_location": "Chicago North Hub, IL",
        "destination": "Chicago, IL",
        "expected_delivery": "2026-09-22 17:00 CST",
        "last_update": "Package loaded onto delivery courier van route 42.",
    },
    "AWB-100200": {
        "awb": "AWB-100200",
        "status": "Exception - Cargo Damage Reported",
        "current_location": "Frankfurt Cargo Center, Germany",
        "destination": "Frankfurt, Germany",
        "expected_delivery": "On Hold",
        "last_update": "Outer shipping carton damaged during transfer. Preliminary report filed under #CLM-9021.",
    },
}

_CACHED_COLLECTION: Optional[Any] = None


def get_tracking_status(awb: str) -> dict[str, Any]:
    """Fetch mock shipment tracking details for a given Air Waybill (AWB).

    Args:
        awb: The tracking number or Air Waybill identifier.

    Returns:
        dict: A dictionary containing live shipment status, location,
            and estimated delivery details.
    """
    clean_awb = awb.strip().upper()
    # Normalize by checking direct key or sanitized key without dashes/spaces
    for key, data in SAMPLE_TRACKING_DATABASE.items():
        if key.upper() == clean_awb or key.replace("-", "").upper() == clean_awb.replace("-", ""):
            return data

    return {
        "awb": awb,
        "status": "Not Found",
        "details": f"No active tracking records found for AWB '{awb}'. Please verify the tracking number.",
    }


def _chunk_document(text: str, filename: str) -> list[dict[str, Any]]:
    """Split a policy document into logical sections or paragraphs."""
    raw_sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    chunks: list[dict[str, Any]] = []

    for i, section in enumerate(raw_sections):
        chunks.append({
            "id": f"{filename}_chunk_{i}",
            "text": section,
            "metadata": {"source": filename, "chunk_index": i},
        })
    return chunks


def setup_chroma_db(
    docs_dir: Optional[str | Path] = None,
    collection_name: str = "logistics_policies",
) -> Any:
    """Initialize ChromaDB and load sample policy documents into a collection.

    Loads delay_policy.txt, damage_claims.txt, and faq.txt using
    sentence-transformers embeddings (all-MiniLM-L6-v2).

    Args:
        docs_dir: Directory containing the policy text files. If None,
            defaults to a 'policies' folder adjacent to this file.
        collection_name: Name of the ChromaDB collection.

    Returns:
        chromadb.Collection: The initialized ChromaDB collection.
    """
    global _CACHED_COLLECTION

    if docs_dir is None:
        docs_dir = Path(__file__).parent / "policies"
    else:
        docs_dir = Path(docs_dir)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    target_files = ["delay_policy.txt", "damage_claims.txt", "faq.txt"]
    all_chunks: list[dict[str, Any]] = []

    for fname in target_files:
        file_path = docs_dir / fname
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            all_chunks.extend(_chunk_document(content, fname))

    if all_chunks:
        collection.add(
            ids=[c["id"] for c in all_chunks],
            documents=[c["text"] for c in all_chunks],
            metadatas=[c["metadata"] for c in all_chunks],
        )

    _CACHED_COLLECTION = collection
    return collection


def research_facts(
    category: str,
    email_text: str,
    awb: Optional[str] = None,
    collection: Optional[Any] = None,
    client: Optional[anthropic.Anthropic] = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Retrieve relevant facts to answer a customer's inquiry using RAG and tools.

    1. Searches ChromaDB for the top 3 relevant document chunks.
    2. Calls get_tracking_status if an AWB is provided.
    3. Sends retrieved facts and live data to Claude to extract exact, verified facts.

    Args:
        category: Email classification category (e.g. 'delay_complaint').
        email_text: Raw customer email text.
        awb: Optional AWB tracking number if mentioned in the email.
        collection: Optional ChromaDB collection. If None, uses cached/initialized collection.
        client: Optional Anthropic API client. If None, initializes default client.
        max_retries: Max retries for handling malformed JSON responses.

    Returns:
        dict: A dictionary containing:
            - facts (list[dict[str, str]]): List of objects each with "fact" and "source".
            - enough_facts_found (bool): Whether sufficient factual basis was gathered.

    Raises:
        ValueError: If unable to parse a valid JSON response within max_retries.
        anthropic.APIError: If the Anthropic API call fails.
    """
    global _CACHED_COLLECTION

    if collection is None:
        if _CACHED_COLLECTION is None:
            setup_chroma_db()
        collection = _CACHED_COLLECTION

    if client is None:
        client = anthropic.Anthropic()

    # 1. Search ChromaDB for top 3 relevant chunks
    search_query = f"Category: {category}. Inquiry: {email_text}"
    query_results = collection.query(
        query_texts=[search_query],
        n_results=3,
    )

    retrieved_chunks: list[str] = []
    if query_results and "documents" in query_results and query_results["documents"]:
        docs = query_results["documents"][0]
        metas = query_results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            source = meta.get("source", "policy_doc") if meta else "policy_doc"
            retrieved_chunks.append(f"[Source: {source}]\n{doc}")

    # 2. Call get_tracking_status if AWB is present
    tracking_info: Optional[dict[str, Any]] = None
    if awb and awb.strip():
        tracking_info = get_tracking_status(awb.strip())

    # Build prompt content for Claude
    sources_text_blocks = []
    if tracking_info:
        sources_text_blocks.append(
            f"LIVE TRACKING TOOL DATA (source: 'live_tracking_tool'):\n"
            f"{json.dumps(tracking_info, indent=2)}"
        )
    else:
        sources_text_blocks.append("LIVE TRACKING TOOL DATA: No AWB provided or searched.")

    sources_text_blocks.append(
        "RETRIEVED STATIC POLICY DOCUMENTS:\n"
        + ("\n\n".join(retrieved_chunks) if retrieved_chunks else "None retrieved.")
    )

    user_prompt = (
        f"CUSTOMER EMAIL:\n"
        f"Category: {category}\n"
        f"Text:\n{email_text}\n\n"
        f"AVAILABLE INFORMATION SOURCES:\n"
        f"{'\n\n'.join(sources_text_blocks)}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt}
    ]

    attempts = 0
    last_parse_error: Optional[Exception] = None

    while attempts <= max_retries:
        attempts += 1
        response = client.messages.create(
            model=MODEL_NAME,
            system=RAG_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )

        raw_output = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                raw_output += block.text
            elif hasattr(block, "text"):
                raw_output += str(block.text)

        try:
            parsed = _extract_and_parse_json(raw_output)
            return parsed
        except ValueError as err:
            last_parse_error = err
            if attempts <= max_retries:
                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Please output ONLY a valid raw JSON object conforming strictly "
                            "to the requested schema: {'facts': [{'fact': str, 'source': str}], 'enough_facts_found': bool}"
                        ),
                    }
                )

    raise ValueError(
        f"Failed to obtain valid JSON from model after {attempts} attempts. "
        f"Last error: {last_parse_error}"
    )


if __name__ == "__main__":
    import pprint

    print("=" * 70)
    print("Testing RAG Agent on Logistics Inquiries")
    print("=" * 70)

    # Setup ChromaDB collection with policies
    print("Initializing ChromaDB collection with policy documents...")
    collection = setup_chroma_db()
    print(f"Loaded ChromaDB collection '{collection.name}' with {collection.count()} chunks.\n")

    test_inquiries = [
        {
            "name": "Case 1: Delayed Shipment with AWB",
            "category": "delay_complaint",
            "awb": "AWB-8849201",
            "email_text": "Shipment AWB-8849201 was supposed to be in Dallas 3 days ago. Why is it delayed and what compensation am I entitled to?",
        },
        {
            "name": "Case 2: Damage Claim without AWB",
            "category": "damage_or_loss_claim",
            "awb": None,
            "email_text": "One of our delivered cartons was crushed on arrival. What is your claim deadline and what proof do I need to send?",
        },
    ]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[NOTE] ANTHROPIC_API_KEY environment variable is not set.")
        print("To run live API calls against Claude, set your API key:")
        print("    $env:ANTHROPIC_API_KEY=\"your-api-key-here\" (PowerShell)\n")
        print("Demonstrating mock outputs with retrieved ChromaDB & tracking facts:\n")

        class MockContent:
            def __init__(self, text: str):
                self.type = "text"
                self.text = text

        class MockResponse:
            def __init__(self, text: str):
                self.content = [MockContent(text)]

        mock_responses = [
            json.dumps({
                "facts": [
                    {
                        "fact": "AWB-8849201 is currently held at Kansas City Hub, MO due to a mechanical delay on the rail line and rescheduled for priority road transfer.",
                        "source": "live_tracking_tool",
                    },
                    {
                        "fact": "New expected delivery date for AWB-8849201 is 2026-09-24 14:00 CST.",
                        "source": "live_tracking_tool",
                    },
                    {
                        "fact": "If an express shipment is delayed by more than 48 hours due to carrier error, consignees are entitled to a 20% freight credit.",
                        "source": "delay_policy.txt",
                    },
                    {
                        "fact": "Delay complaints must be formally submitted within 14 business days of the original expected delivery date.",
                        "source": "delay_policy.txt",
                    },
                ],
                "enough_facts_found": True,
            }),
            json.dumps({
                "facts": [
                    {
                        "fact": "Formal cargo damage claims must be filed within 7 calendar days of package receipt.",
                        "source": "damage_claims.txt",
                    },
                    {
                        "fact": "Claimants must provide photos of outer packaging with legible shipping label, inner protective packaging, and damage to cargo, along with the original vendor commercial invoice.",
                        "source": "damage_claims.txt",
                    },
                    {
                        "fact": "Standard carrier liability coverage is capped at $100.00 USD unless Declared Value Insurance was contracted.",
                        "source": "damage_claims.txt",
                    },
                ],
                "enough_facts_found": True,
            }),
        ]

        for i, inq in enumerate(test_inquiries):
            print(f"--- {inq['name']} ---")
            print(f"Category: {inq['category']} | AWB: {inq['awb']}")
            print(f"Email: {inq['email_text']}")

            mock_client = anthropic.Anthropic(api_key="mock-key")
            mock_client.messages.create = lambda *args, **kwargs: MockResponse(mock_responses[i])  # type: ignore[method-assign]
            result = research_facts(
                category=inq["category"],
                email_text=inq["email_text"],
                awb=inq["awb"],
                collection=collection,
                client=mock_client,
            )
            print("\nExtracted Facts:")
            pprint.pprint(result)
            print()
    else:
        for inq in test_inquiries:
            print(f"--- {inq['name']} ---")
            print(f"Category: {inq['category']} | AWB: {inq['awb']}")
            print(f"Email: {inq['email_text']}")
            try:
                result = research_facts(
                    category=inq["category"],
                    email_text=inq["email_text"],
                    awb=inq["awb"],
                    collection=collection,
                )
                print("\nExtracted Facts:")
                pprint.pprint(result)
            except Exception as e:
                print(f"Error calling Claude API: {e}")
            print()
