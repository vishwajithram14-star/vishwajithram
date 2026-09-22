"""Priority Agent for logistics email support system.

This module scores the urgency and priority of logistics email requests based on
the output of classifier_agent.py and the customer's account tier.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import anthropic

from classifier_agent import MODEL_NAME, _extract_and_parse_json

PRIORITY_SYSTEM_PROMPT = """You are the Priority Agent for a logistics company's email support system.
Score urgency 0-100 using: goods type (0-25), delay length (0-25), urgent 
language (0-20), customer tier (0-15), money/penalty risk (0-15).
Return ONLY JSON: {"priority": "P1|P2|P3|P4", "score": int, 
"reply_deadline": str, "reasons": [str]}.
P1 >=75 (15 min), P2 50-74 (1hr), P3 25-49 (4hr), P4 <25 (24hr)."""


def score_priority(
    classifier_output: dict[str, Any],
    customer_tier: str,
    client: Optional[anthropic.Anthropic] = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Score the urgency and support priority of a classified email.

    Args:
        classifier_output: Dictionary output from classify_email containing
            category, intent, awb, sentiment, language, and confidence.
        customer_tier: The tier of the customer (e.g. "Platinum", "Gold", "Standard").
        client: Optional pre-configured Anthropic client instance. If None,
            a default `anthropic.Anthropic()` client is initialized.
        max_retries: Maximum number of retries if JSON parsing fails. Default is 1.

    Returns:
        dict: A dictionary containing:
            - priority (str): One of "P1", "P2", "P3", "P4".
            - score (int): Urgency score between 0 and 100.
            - reply_deadline (str): Response target (e.g. "15 min", "1hr", "4hr", "24hr").
            - reasons (list[str]): Factors contributing to the assigned score.

    Raises:
        ValueError: If unable to parse a valid JSON response within max_retries.
        anthropic.APIError: If the Anthropic API call fails.
    """
    if client is None:
        client = anthropic.Anthropic()

    user_payload = {
        "classifier_output": classifier_output,
        "customer_tier": customer_tier,
    }
    user_message = json.dumps(user_payload, indent=2)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message}
    ]

    attempts = 0
    last_parse_error: Optional[Exception] = None

    while attempts <= max_retries:
        attempts += 1
        response = client.messages.create(
            model=MODEL_NAME,
            system=PRIORITY_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )

        # Concatenate text content blocks from response
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
                            "to the requested schema with no surrounding text or formatting."
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
    print("Testing Priority Agent on Sample Classified Cases")
    print("=" * 70)

    sample_cases = [
        {
            "name": "Case 1: Severe Delay + Halted Production (Platinum Tier)",
            "tier": "Platinum",
            "classifier_output": {
                "category": "delay_complaint",
                "intent": "demand immediate location and delivery update for halted production cargo",
                "awb": "AWB-8849201",
                "sentiment": "angry",
                "language": "en",
                "confidence": 0.98,
            },
        },
        {
            "name": "Case 2: Standard Status Check (Gold Tier)",
            "tier": "Gold",
            "classifier_output": {
                "category": "tracking_query",
                "intent": "request shipment status and customs clearance ETA",
                "awb": "551-09827461",
                "sentiment": "neutral",
                "language": "en",
                "confidence": 0.95,
            },
        },
        {
            "name": "Case 3: Non-urgent Social Email (Standard Tier)",
            "tier": "Standard",
            "classifier_output": {
                "category": "other",
                "intent": "informal social greeting unrelated to logistics support",
                "awb": None,
                "sentiment": "neutral",
                "language": "en",
                "confidence": 0.25,
            },
        },
    ]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[NOTE] ANTHROPIC_API_KEY environment variable is not set.")
        print("To run live API calls against Claude, set your API key:")
        print("    $env:ANTHROPIC_API_KEY=\"your-api-key-here\" (PowerShell)")
        print("    set ANTHROPIC_API_KEY=your-api-key-here (CMD)\n")
        print("Simulating sample outputs with mock response demonstration:\n")

        class MockContent:
            def __init__(self, text: str):
                self.type = "text"
                self.text = text

        class MockResponse:
            def __init__(self, text: str):
                self.content = [MockContent(text)]

        mock_responses = [
            json.dumps({
                "priority": "P1",
                "score": 92,
                "reply_deadline": "15 min",
                "reasons": [
                    "Factory production halted causing financial loss (penalty risk 15/15)",
                    "Severe shipment delay spanning several days (23/25)",
                    "Platinum tier customer escalation (15/15)",
                    "Angry and urgent customer sentiment (19/20)",
                    "Critical manufacturing goods (20/25)",
                ],
            }),
            json.dumps({
                "priority": "P3",
                "score": 42,
                "reply_deadline": "4hr",
                "reasons": [
                    "Routine tracking and customs query with no reported damage (10/25)",
                    "No active critical delay reported (5/25)",
                    "Gold tier customer (10/15)",
                    "Neutral sentiment (5/20)",
                    "Low immediate financial risk (12/15)",
                ],
            }),
            json.dumps({
                "priority": "P4",
                "score": 10,
                "reply_deadline": "24hr",
                "reasons": [
                    "Unrelated non-operational social greeting (0/25)",
                    "Standard tier customer (3/15)",
                    "Neutral sentiment (2/20)",
                    "Zero financial or delay risk (5/15)",
                ],
            }),
        ]

        for i, case in enumerate(sample_cases):
            print(f"--- {case['name']} ---")
            print("Customer Tier:", case["tier"])
            print("Classifier Output:")
            pprint.pprint(case["classifier_output"])

            mock_client = anthropic.Anthropic(api_key="mock-key")
            mock_client.messages.create = lambda *args, **kwargs: MockResponse(mock_responses[i])  # type: ignore[method-assign]
            result = score_priority(
                classifier_output=case["classifier_output"],
                customer_tier=case["tier"],
                client=mock_client,
            )
            print("\nPriority Output:")
            pprint.pprint(result)
            print()
    else:
        for case in sample_cases:
            print(f"--- {case['name']} ---")
            print("Customer Tier:", case["tier"])
            print("Classifier Output:")
            pprint.pprint(case["classifier_output"])
            try:
                result = score_priority(
                    classifier_output=case["classifier_output"],
                    customer_tier=case["tier"],
                )
                print("\nPriority Output:")
                pprint.pprint(result)
            except Exception as e:
                print(f"Error calling Claude API: {e}")
            print()
