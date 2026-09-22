"""Draft/Action Agent for logistics email support system.

This module generates professional, polite customer email responses using ONLY
the verified facts gathered by previous agents and proposes operational actions
from an approved whitelist.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import anthropic

from classifier_agent import MODEL_NAME, _extract_and_parse_json

# Exact system prompt for Draft/Action Agent
DRAFT_SYSTEM_PROMPT = """You are the Draft/Action Agent for a logistics company's email support system.
Write a reply using ONLY the provided facts, polite and calm. Propose actions 
only from: ["reply_only", "create_ticket", "reschedule_delivery", 
"raise_damage_claim", "request_refund_review", "escalate_to_manager"].
Return ONLY JSON: {"draft_reply": str, "proposed_actions": [str], 
"facts_used": [str]}.
Never promise a refund/discount amount yourself."""

ALLOWED_ACTIONS = {
    "reply_only",
    "create_ticket",
    "reschedule_delivery",
    "raise_damage_claim",
    "request_refund_review",
    "escalate_to_manager",
}

REFUND_AMOUNT_PATTERN = re.compile(
    r"(\$\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*%\s*(?:refund|discount|credit))",
    re.IGNORECASE,
)


def draft_reply(
    email_text: str,
    facts: list[dict[str, Any]],
    client: Optional[anthropic.Anthropic] = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Generate a customer reply and proposed operational actions based strictly on verified facts.

    Args:
        email_text: Raw incoming customer email text.
        facts: List of fact objects, typically formatted as [{'fact': str, 'source': str}].
        client: Optional pre-configured Anthropic client instance. If None,
            a default `anthropic.Anthropic()` client is initialized.
        max_retries: Maximum number of retries if JSON parsing fails. Default is 1.

    Returns:
        dict: A dictionary containing:
            - draft_reply (str): Customer-facing polite reply based only on facts.
            - proposed_actions (list[str]): Proposed actions from the approved whitelist.
            - facts_used (list[str]): List of facts cited in the reply.

    Raises:
        ValueError: If unable to parse a valid JSON response within max_retries
            or if invalid proposed actions are returned.
        anthropic.APIError: If the Anthropic API call fails.
    """
    if client is None:
        client = anthropic.Anthropic()

    facts_payload = json.dumps(facts, indent=2)
    user_prompt = (
        f"CUSTOMER EMAIL:\n"
        f"{email_text}\n\n"
        f"VERIFIED FACTS AVAILABLE:\n"
        f"{facts_payload}\n\n"
        f"Please compose the customer response and propose applicable actions "
        f"conforming strictly to your instructions."
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
            system=DRAFT_SYSTEM_PROMPT,
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

            # Validate schema
            if not isinstance(parsed, dict) or "draft_reply" not in parsed:
                raise ValueError("Parsed output does not contain 'draft_reply' field.")

            # Filter or sanitize proposed actions against allowed list
            if "proposed_actions" in parsed and isinstance(parsed["proposed_actions"], list):
                parsed["proposed_actions"] = [
                    action for action in parsed["proposed_actions"]
                    if action in ALLOWED_ACTIONS
                ]
            else:
                parsed["proposed_actions"] = ["reply_only"]

            if "facts_used" not in parsed or not isinstance(parsed["facts_used"], list):
                parsed["facts_used"] = []

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
                            "Please output ONLY a valid raw JSON object conforming strictly to: "
                            '{"draft_reply": str, "proposed_actions": [str], "facts_used": [str]}'
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
    print("Testing Draft/Action Agent on Logistics Inquiries")
    print("=" * 70)

    sample_cases = [
        {
            "name": "Case 1: Delayed Shipment with Refund Inquiry",
            "email_text": (
                "Subject: URGENT: Delivery overdue for shipment AWB-8849201\n\n"
                "Our production line is halted because shipment AWB-8849201 was supposed to arrive "
                "in Dallas 3 days ago. Where is our cargo and what compensation are you giving us?"
            ),
            "facts": [
                {
                    "fact": "AWB-8849201 is currently held at Kansas City Hub, MO due to a mechanical delay on the rail freight line.",
                    "source": "live_tracking_tool",
                },
                {
                    "fact": "Cargo has been rescheduled for priority road transfer with an updated delivery estimate of 2026-09-24 14:00 CST.",
                    "source": "live_tracking_tool",
                },
                {
                    "fact": "Express shipments delayed over 48 hours due to carrier error are eligible for credit review upon formal claim submission within 14 business days.",
                    "source": "delay_policy.txt",
                },
            ],
        },
        {
            "name": "Case 2: Cargo Damage Report",
            "email_text": (
                "Subject: Damaged container on delivery\n\n"
                "Our shipment AWB-100200 arrived with torn packaging and physical item damage. "
                "Please advise how we can claim our losses."
            ),
            "facts": [
                {
                    "fact": "AWB-100200 status is Exception - Cargo Damage Reported at Frankfurt Cargo Center under claim ticket #CLM-9021.",
                    "source": "live_tracking_tool",
                },
                {
                    "fact": "Damage claims must be submitted within 7 calendar days of package receipt.",
                    "source": "damage_claims.txt",
                },
                {
                    "fact": "Claimants must provide photos of outer packaging, inner packaging, damaged cargo, and the original vendor commercial invoice.",
                    "source": "damage_claims.txt",
                },
            ],
        },
    ]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[NOTE] ANTHROPIC_API_KEY environment variable is not set.")
        print("To run live API calls against Claude, set your API key:")
        print("    $env:ANTHROPIC_API_KEY=\"your-api-key-here\" (PowerShell)\n")
        print("Demonstrating mock outputs with factual draft responses:\n")

        class MockContent:
            def __init__(self, text: str):
                self.type = "text"
                self.text = text

        class MockResponse:
            def __init__(self, text: str):
                self.content = [MockContent(text)]

        mock_responses = [
            json.dumps({
                "draft_reply": (
                    "Dear Customer,\n\n"
                    "Thank you for contacting us regarding shipment AWB-8849201. "
                    "We sincerely apologize for the delay. Your cargo is currently at the Kansas City Hub, MO, "
                    "following a mechanical delay on the rail freight line. It has been rescheduled for priority "
                    "road transfer, with delivery now expected on September 24, 2026 at 14:00 CST.\n\n"
                    "Regarding compensation, shipments delayed beyond 48 hours are eligible for a credit review "
                    "when submitted within 14 business days. I have initiated a refund review request with our billing team "
                    "who will evaluate your shipment details and contact you directly.\n\n"
                    "Best regards,\nCustomer Support Team"
                ),
                "proposed_actions": ["reschedule_delivery", "request_refund_review"],
                "facts_used": [
                    "AWB-8849201 is held at Kansas City Hub due to rail mechanical delay",
                    "Rescheduled for priority road transfer with delivery on 2026-09-24 14:00 CST",
                    "Delay credit eligibility requires review and filing within 14 business days",
                ],
            }),
            json.dumps({
                "draft_reply": (
                    "Dear Customer,\n\n"
                    "We are sorry to hear that your shipment arrived damaged. "
                    "A damage incident has been registered under claim ticket #CLM-9021 at our Frankfurt Cargo Center.\n\n"
                    "To proceed with your claim, please submit your formal claim within 7 calendar days of delivery. "
                    "Kindly provide photographs of the outer packaging (with shipping label visible), inner protective packaging, "
                    "the damaged items, and the original vendor commercial invoice.\n\n"
                    "We have opened a formal damage claim on your behalf.\n\n"
                    "Sincerely,\nCargo Claims Department"
                ),
                "proposed_actions": ["raise_damage_claim", "create_ticket"],
                "facts_used": [
                    "Status is Exception - Cargo Damage Reported under claim ticket #CLM-9021",
                    "Damage claims must be submitted within 7 calendar days",
                    "Photos of packaging, damaged cargo, and commercial invoice are required",
                ],
            }),
        ]

        for i, case in enumerate(sample_cases):
            print(f"--- {case['name']} ---")
            print("Customer Email:\n", case["email_text"].strip())
            print("\nProvided Facts:")
            pprint.pprint(case["facts"])

            mock_client = anthropic.Anthropic(api_key="mock-key")
            mock_client.messages.create = lambda *args, **kwargs: MockResponse(mock_responses[i])  # type: ignore[method-assign]
            result = draft_reply(
                email_text=case["email_text"],
                facts=case["facts"],
                client=mock_client,
            )
            print("\nGenerated Draft & Actions:")
            pprint.pprint(result)
            print()
    else:
        for case in sample_cases:
            print(f"--- {case['name']} ---")
            print("Customer Email:\n", case["email_text"].strip())
            try:
                result = draft_reply(
                    email_text=case["email_text"],
                    facts=case["facts"],
                )
                print("\nGenerated Draft & Actions:")
                pprint.pprint(result)
            except Exception as e:
                print(f"Error calling Claude API: {e}")
            print()
