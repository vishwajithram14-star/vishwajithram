"""Orchestrator for logistics email support multi-agent system.

Combines classifier_agent, priority_agent, rag_agent, draft_agent, and guard_agent
into an automated pipeline using LangGraph, incorporating error handling,
deterministic conditional routing, and retry loops.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, TypedDict

import anthropic
from langgraph.graph import END, StateGraph

from classifier_agent import classify_email
from draft_agent import draft_reply
from guard_agent import review_draft
from priority_agent import score_priority
from rag_agent import research_facts, setup_chroma_db

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class EmailState(TypedDict, total=False):
    """Complete state dictionary flowing through the LangGraph email support pipeline."""
    # Input fields
    email_text: str
    customer_tier: str

    # Classifier agent outputs
    category: Optional[str]
    intent: Optional[str]
    awb: Optional[str]
    sentiment: Optional[str]
    confidence: Optional[float]

    # Priority agent outputs
    priority: Optional[str]
    score: Optional[int]
    reply_deadline: Optional[str]

    # RAG agent outputs
    facts: Optional[list[dict[str, Any]]]
    enough_facts_found: Optional[bool]

    # Draft agent outputs
    draft_reply: Optional[str]
    proposed_actions: Optional[list[str]]

    # Guard agent outputs
    guard_verdict: Optional[str]
    guard_notes: Optional[str]

    # Routing & control fields
    retry_count: int
    route: Optional[str]

    # Execution context (optional client injection)
    client: Optional[Any]


# ============================================================================
# Smart Mock Fallback Client (Active when ANTHROPIC_API_KEY is not set)
# ============================================================================

class _MockContentBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _MockResponse:
    def __init__(self, text: str):
        self.content = [_MockContentBlock(text)]


class _DemoMockClient:
    """Mock client providing realistic responses for demonstration when no API key is set."""

    def __init__(self):
        self.messages = self

    def create(self, model: str, system: str, messages: list[dict[str, Any]], **kwargs) -> _MockResponse:
        user_content = messages[-1].get("content", "")

        # 1. Classifier Agent
        if "You are the Classifier Agent" in system:
            content_lower = user_content.lower()
            if "ignore all previous" in content_lower or "system instruction override" in content_lower:
                return _MockResponse(json.dumps({
                    "category": "other",
                    "intent": "prompt injection test / administrative override request",
                    "awb": None,
                    "sentiment": "neutral",
                    "language": "en",
                    "confidence": 0.20,
                }))
            elif "factory down" in content_lower or "awb-8849201" in content_lower or "overdue" in content_lower:
                return _MockResponse(json.dumps({
                    "category": "delay_complaint",
                    "intent": "demand urgent status and compensation for overdue manufacturing cargo",
                    "awb": "AWB-8849201",
                    "sentiment": "angry",
                    "language": "en",
                    "confidence": 0.98,
                }))
            elif "damaged" in content_lower or "awb-100200" in content_lower or "crushed" in content_lower:
                return _MockResponse(json.dumps({
                    "category": "damage_or_loss_claim",
                    "intent": "report transit damage and request loss claim procedures",
                    "awb": "AWB-100200",
                    "sentiment": "neutral",
                    "language": "en",
                    "confidence": 0.96,
                }))
            elif "quote" in content_lower or "booking" in content_lower or "pallets" in content_lower:
                return _MockResponse(json.dumps({
                    "category": "booking_or_quote_request",
                    "intent": "request freight rate quote and booking availability for 5 pallets",
                    "awb": None,
                    "sentiment": "neutral",
                    "language": "en",
                    "confidence": 0.92,
                }))
            else:
                return _MockResponse(json.dumps({
                    "category": "documents_or_customs",
                    "intent": "inquire about customs clearance timeline and terminal storage fees",
                    "awb": None,
                    "sentiment": "neutral",
                    "language": "en",
                    "confidence": 0.95,
                }))

        # 2. Priority Agent
        elif "You are the Priority Agent" in system:
            try:
                payload = json.loads(user_content)
                cat = payload.get("classifier_output", {}).get("category", "")
                tier = payload.get("customer_tier", "Standard")
            except Exception:
                cat = ""
                tier = "Standard"

            if cat == "delay_complaint" or tier == "Platinum":
                return _MockResponse(json.dumps({
                    "priority": "P1",
                    "score": 92,
                    "reply_deadline": "15 min",
                    "reasons": ["Factory line stopped ($15k/hr loss)", "Platinum tier customer", "Angry sentiment"],
                }))
            elif cat == "damage_or_loss_claim":
                return _MockResponse(json.dumps({
                    "priority": "P2",
                    "score": 65,
                    "reply_deadline": "1hr",
                    "reasons": ["Visible cargo damage claim", "Gold tier customer"],
                }))
            elif cat == "booking_or_quote_request":
                return _MockResponse(json.dumps({
                    "priority": "P3",
                    "score": 40,
                    "reply_deadline": "4hr",
                    "reasons": ["New freight booking quote request", "Standard SLA"],
                }))
            else:
                return _MockResponse(json.dumps({
                    "priority": "P4",
                    "score": 15,
                    "reply_deadline": "24hr",
                    "reasons": ["Standard FAQ inquiry", "No active delay or financial risk"],
                }))

        # 3. RAG Agent
        elif "You are the RAG (Research) Agent" in system:
            content_lower = user_content.lower()
            if "booking_or_quote_request" in content_lower or "5 pallets" in content_lower:
                return _MockResponse(json.dumps({
                    "facts": [],
                    "enough_facts_found": False,
                }))
            elif "awb-8849201" in content_lower:
                return _MockResponse(json.dumps({
                    "facts": [
                        {
                            "fact": "AWB-8849201 is currently held at Kansas City Hub, MO due to a rail line mechanical delay and rescheduled for priority road delivery arriving 2026-09-24 14:00 CST.",
                            "source": "live_tracking_tool",
                        },
                        {
                            "fact": "Express shipments delayed over 48 hours due to carrier error qualify for a 20% freight credit review upon claim within 14 business days.",
                            "source": "delay_policy.txt",
                        },
                    ],
                    "enough_facts_found": True,
                }))
            elif "awb-100200" in content_lower or "damage" in content_lower:
                return _MockResponse(json.dumps({
                    "facts": [
                        {
                            "fact": "Incident registered under claim ticket #CLM-9021 at Frankfurt Cargo Center with cargo damage reported.",
                            "source": "live_tracking_tool",
                        },
                        {
                            "fact": "Formal cargo damage claims must be submitted within 7 calendar days of receipt.",
                            "source": "damage_claims.txt",
                        },
                        {
                            "fact": "Claimants must submit photographs of outer packaging with label legible, inner packaging, damaged items, and the original vendor commercial invoice.",
                            "source": "damage_claims.txt",
                        },
                    ],
                    "enough_facts_found": True,
                }))
            else:
                return _MockResponse(json.dumps({
                    "facts": [
                        {
                            "fact": "Routine customs clearance takes 24 to 48 hours for general cargo.",
                            "source": "faq.txt",
                        },
                        {
                            "fact": "Terminal storage enjoys a 3-day free grace period before demurrage fees of $25.00/day apply.",
                            "source": "faq.txt",
                        },
                    ],
                    "enough_facts_found": True,
                }))

        # 4. Draft Agent
        elif "You are the Draft/Action Agent" in system:
            content_lower = user_content.lower()
            if "awb-8849201" in content_lower:
                return _MockResponse(json.dumps({
                    "draft_reply": (
                        "Dear Mr. Higgins,\n\n"
                        "We sincerely apologize for the disruption caused to your Dallas assembly plant. "
                        "Shipment AWB-8849201 was held at Kansas City Hub due to a rail freight mechanical issue. "
                        "It has been transferred to priority road delivery and is scheduled to arrive on September 24 at 14:00 CST.\n\n"
                        "We have opened an expedited monitoring ticket and submitted a delay credit review request "
                        "with our billing department.\n\n"
                        "Best regards,\nLogistics Operations Management"
                    ),
                    "proposed_actions": ["reschedule_delivery", "request_refund_review", "create_ticket"],
                    "facts_used": [
                        "Held at Kansas City Hub due to rail mechanical delay",
                        "Rescheduled for road delivery arriving 2026-09-24 14:00 CST",
                        "Delay credit review requested under 14-day policy window",
                    ],
                }))
            elif "awb-100200" in content_lower or "damage" in content_lower:
                return _MockResponse(json.dumps({
                    "draft_reply": (
                        "Dear Ms. Dubois,\n\n"
                        "We are very sorry to learn that items in shipment AWB-100200 arrived damaged. "
                        "An initial exception was logged at Frankfurt under incident #CLM-9021.\n\n"
                        "To complete your claim, please submit: (1) photos of the outer shipping box with label visible, "
                        "(2) photos of protective packaging and damaged units, and (3) your original commercial invoice. "
                        "Claims must be submitted within 7 calendar days of receipt.\n\n"
                        "Sincerely,\nCargo Claims Department"
                    ),
                    "proposed_actions": ["raise_damage_claim", "create_ticket"],
                    "facts_used": [
                        "Incident logged under claim #CLM-9021",
                        "7 calendar days claim deadline",
                        "Required photos and commercial invoice",
                    ],
                }))
            else:
                return _MockResponse(json.dumps({
                    "draft_reply": (
                        "Dear Thomas,\n\n"
                        "Thank you for contacting European Logistics Support. "
                        "Standard customs clearance for European air freight typically takes 24 to 48 hours for general cargo. "
                        "Additionally, all cargo receives a 3-day complimentary storage grace period at our airport terminals, "
                        "so no demurrage fees will apply for a 2-day hold.\n\n"
                        "Best regards,\nCustomer Support Team"
                    ),
                    "proposed_actions": ["reply_only"],
                    "facts_used": [
                        "Customs clearance averages 24 to 48 hours",
                        "Terminal storage includes 3 free days before demurrage",
                    ],
                }))

        # 5. Guard Agent
        elif "You are the Guard Agent" in system:
            content_lower = user_content.lower()
            if "ignore all previous" in content_lower or "administrative test mode" in content_lower:
                return _MockResponse(json.dumps({
                    "facts_check": "fail",
                    "privacy_check": "fail",
                    "tone_check": "pass",
                    "policy_check": "fail",
                    "injection_detected": True,
                    "risk_level": "high",
                    "verdict": "FAIL",
                    "notes": "Prompt injection detected in customer input.",
                }))
            elif "awb-8849201" in content_lower or "p1" in user_content:
                return _MockResponse(json.dumps({
                    "facts_check": "pass",
                    "privacy_check": "pass",
                    "tone_check": "pass",
                    "policy_check": "pass",
                    "injection_detected": False,
                    "risk_level": "high",
                    "verdict": "NEEDS_HUMAN",
                    "notes": "Draft is factually sound, but P1 priority critical SLA requires human supervisor review.",
                }))
            elif "damage" in content_lower or "awb-100200" in content_lower or "p2" in user_content:
                return _MockResponse(json.dumps({
                    "facts_check": "pass",
                    "privacy_check": "pass",
                    "tone_check": "pass",
                    "policy_check": "pass",
                    "injection_detected": False,
                    "risk_level": "medium",
                    "verdict": "NEEDS_HUMAN",
                    "notes": "Cargo damage claim involves potential carrier liability. Requires human approval.",
                }))
            else:
                return _MockResponse(json.dumps({
                    "facts_check": "pass",
                    "privacy_check": "pass",
                    "tone_check": "pass",
                    "policy_check": "pass",
                    "injection_detected": False,
                    "risk_level": "low",
                    "verdict": "SAFE",
                    "notes": "Low-risk FAQ response. Facts match documentation and tone is courteous.",
                }))

        return _MockResponse(json.dumps({"status": "ok"}))


def _resolve_client(state: EmailState) -> Any:
    """Return an injected client, a live Anthropic client if key is set, or a demo mock client."""
    if state.get("client") is not None:
        return state["client"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()
    return _DemoMockClient()


# ============================================================================
# LangGraph Agent Nodes
# ============================================================================

def classify_node(state: EmailState) -> EmailState:
    """Node wrapping classifier_agent.py."""
    logger.info("Executing classify_node...")
    client = _resolve_client(state)
    try:
        result = classify_email(state["email_text"], client=client)
        state["category"] = result.get("category")
        state["intent"] = result.get("intent")
        state["awb"] = result.get("awb")
        state["sentiment"] = result.get("sentiment")
        state["confidence"] = float(result.get("confidence", 0.0))
        logger.info(f"Classified category='{state['category']}', confidence={state['confidence']}")
    except Exception as err:
        logger.error(f"Error in classify_node: {err}", exc_info=True)
        state["route"] = "human_triage"
        state["guard_notes"] = f"Classifier error: {err}"
    return state


def priority_node(state: EmailState) -> EmailState:
    """Node wrapping priority_agent.py."""
    logger.info("Executing priority_node...")
    client = _resolve_client(state)
    try:
        classifier_output = {
            "category": state.get("category", "other"),
            "intent": state.get("intent", ""),
            "awb": state.get("awb"),
            "sentiment": state.get("sentiment", "neutral"),
            "confidence": state.get("confidence", 0.0),
        }
        tier = state.get("customer_tier", "Standard")
        result = score_priority(classifier_output, customer_tier=tier, client=client)
        state["priority"] = result.get("priority")
        state["score"] = int(result.get("score", 0))
        state["reply_deadline"] = result.get("reply_deadline")
        logger.info(f"Priority scored priority='{state['priority']}', deadline='{state['reply_deadline']}'")
    except Exception as err:
        logger.error(f"Error in priority_node: {err}", exc_info=True)
        state["route"] = "human_triage"
        state["guard_notes"] = f"Priority scoring error: {err}"
    return state


def rag_node(state: EmailState) -> EmailState:
    """Node wrapping rag_agent.py."""
    logger.info("Executing rag_node...")
    client = _resolve_client(state)
    try:
        result = research_facts(
            category=state.get("category", "other"),
            email_text=state["email_text"],
            awb=state.get("awb"),
            client=client,
        )
        state["facts"] = result.get("facts", [])
        state["enough_facts_found"] = bool(result.get("enough_facts_found", False))
        logger.info(f"RAG facts found={len(state['facts'])}, enough={state['enough_facts_found']}")
    except Exception as err:
        logger.error(f"Error in rag_node: {err}", exc_info=True)
        state["route"] = "human_triage"
        state["guard_notes"] = f"RAG research error: {err}"
    return state


def draft_node(state: EmailState) -> EmailState:
    """Node wrapping draft_agent.py."""
    logger.info("Executing draft_node...")
    client = _resolve_client(state)
    try:
        facts = state.get("facts") or []
        result = draft_reply(
            email_text=state["email_text"],
            facts=facts,
            client=client,
        )
        state["draft_reply"] = result.get("draft_reply", "")
        state["proposed_actions"] = result.get("proposed_actions", ["reply_only"])
        logger.info(f"Draft generated ({len(state['draft_reply'])} chars), actions={state['proposed_actions']}")
    except Exception as err:
        logger.error(f"Error in draft_node: {err}", exc_info=True)
        state["route"] = "human_triage"
        state["guard_notes"] = f"Drafting error: {err}"
    return state


def guard_node(state: EmailState) -> EmailState:
    """Node wrapping guard_agent.py."""
    logger.info("Executing guard_node...")
    client = _resolve_client(state)
    try:
        policy_limits = {
            "priority": state.get("priority", "P3"),
            "max_auto_refund": 0.0,
        }
        result = review_draft(
            email_text=state["email_text"],
            draft_reply=state.get("draft_reply", ""),
            facts=state.get("facts") or [],
            policy_limits=policy_limits,
            client=client,
        )
        state["guard_verdict"] = result.get("verdict", "FAIL")
        state["guard_notes"] = result.get("notes", "")
        logger.info(f"Guard review verdict='{state['guard_verdict']}'")
    except Exception as err:
        logger.error(f"Error in guard_node: {err}", exc_info=True)
        state["route"] = "human_triage"
        state["guard_notes"] = f"Guard audit error: {err}"
    return state


# ============================================================================
# Routing Nodes (No LLM Call)
# ============================================================================

def human_triage(state: EmailState) -> EmailState:
    """Route to human agent when classification is uncertain or an error occurred."""
    logger.info("Routing -> human_triage")
    state["route"] = "human_triage"
    return state


def escalate(state: EmailState) -> EmailState:
    """Escalate to specialist team when insufficient ground facts are available."""
    logger.info("Routing -> escalate")
    state["route"] = "escalate"
    return state


def auto_send(state: EmailState) -> EmailState:
    """Automatically send verified safe draft response to customer."""
    logger.info("Routing -> auto_send")
    state["route"] = "auto_send"
    return state


def human_approval(state: EmailState) -> EmailState:
    """Send draft to customer support supervisor for human sign-off before dispatch."""
    logger.info("Routing -> human_approval")
    state["route"] = "human_approval"
    return state


def block_retry(state: EmailState) -> EmailState:
    """Reject non-compliant draft, increment retry counter, and loop back to draft agent."""
    logger.info(f"Routing -> block_retry (attempt {state.get('retry_count', 0) + 1})")
    state["route"] = "block_retry"
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


# ============================================================================
# Conditional Edge Routers
# ============================================================================

def route_after_classifier(state: EmailState) -> str:
    """Branch from classifier: if confidence >= 0.75: priority, else: human_triage."""
    if state.get("route") == "human_triage":
        return "human_triage"
    confidence = float(state.get("confidence", 0.0))
    if confidence >= 0.75:
        return "priority"
    return "human_triage"


def route_after_rag(state: EmailState) -> str:
    """Branch from RAG: if enough_facts_found: draft, else: escalate."""
    if state.get("route") == "human_triage":
        return "human_triage"
    if state.get("enough_facts_found", False):
        return "draft"
    return "escalate"


def route_after_guard(state: EmailState) -> str:
    """Branch from Guard:
    SAFE -> auto_send
    NEEDS_HUMAN -> human_approval
    FAIL and retry_count < 2 -> block_retry
    FAIL and retry_count >= 2 -> human_approval
    """
    if state.get("route") == "human_triage":
        return "human_triage"

    verdict = str(state.get("guard_verdict", "FAIL")).upper()
    retries = int(state.get("retry_count", 0))

    if verdict == "SAFE":
        return "auto_send"
    elif verdict == "NEEDS_HUMAN":
        return "human_approval"
    elif verdict == "FAIL":
        if retries < 2:
            return "block_retry"
        else:
            return "human_approval"
    return "human_approval"


# ============================================================================
# LangGraph Workflow Construction
# ============================================================================

def build_email_support_graph() -> Any:
    """Construct and compile the LangGraph workflow for email triage and response."""
    workflow = StateGraph(EmailState)

    # 1. Register LLM Agent Nodes
    workflow.add_node("classifier", classify_node)
    workflow.add_node("priority", priority_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("guard", guard_node)

    # 2. Register Non-LLM Routing Nodes
    workflow.add_node("human_triage", human_triage)
    workflow.add_node("escalate", escalate)
    workflow.add_node("auto_send", auto_send)
    workflow.add_node("human_approval", human_approval)
    workflow.add_node("block_retry", block_retry)

    # 3. Entry Point
    workflow.set_entry_point("classifier")

    # 4. Define Transitions
    # Classifier -> [if confidence >= 0.75: priority, else: human_triage]
    workflow.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "priority": "priority",
            "human_triage": "human_triage",
        },
    )

    # Priority -> RAG
    workflow.add_edge("priority", "rag")

    # RAG -> [if enough_facts_found: draft, else: escalate]
    workflow.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "draft": "draft",
            "escalate": "escalate",
            "human_triage": "human_triage",
        },
    )

    # Draft -> Guard
    workflow.add_edge("draft", "guard")

    # Guard -> [SAFE: auto_send, NEEDS_HUMAN: human_approval, FAIL and retry < 2: block_retry, FAIL and retry >= 2: human_approval]
    workflow.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "auto_send": "auto_send",
            "human_approval": "human_approval",
            "block_retry": "block_retry",
            "human_triage": "human_triage",
        },
    )

    # block_retry -> draft (loop back)
    workflow.add_edge("block_retry", "draft")

    # Terminal nodes
    workflow.add_edge("human_triage", END)
    workflow.add_edge("escalate", END)
    workflow.add_edge("auto_send", END)
    workflow.add_edge("human_approval", END)

    return workflow.compile()


app = build_email_support_graph()
