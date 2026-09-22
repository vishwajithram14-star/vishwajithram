"""Run demonstration of the LangGraph Logistics Email Support Multi-Agent System.

Loads sample emails from data/sample_emails.json, executes each through
app.invoke(), and prints a clean, structured summary table.
"""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator import app


def run_demo() -> list[dict]:
    """Execute all sample emails through the compiled LangGraph pipeline."""
    sample_file = Path(__file__).parent / "data" / "sample_emails.json"
    if not sample_file.exists():
        raise FileNotFoundError(f"Cannot find sample emails at {sample_file}")

    emails = json.loads(sample_file.read_text(encoding="utf-8"))
    results = []

    print("=" * 140)
    print(f"PROCESSING {len(emails)} SAMPLE EMAILS THROUGH LANGGRAPH ORCHESTRATOR")
    print("=" * 140)

    for i, item in enumerate(emails, 1):
        subject = item.get("subject", "No Subject")
        text = item.get("email_text", "")
        tier = item.get("customer_tier", "Standard")

        print(f"\n[{i}/{len(emails)}] Invoking Workflow for: '{subject}' (Tier: {tier})...")

        initial_state = {
            "email_text": text,
            "customer_tier": tier,
            "retry_count": 0,
        }

        final_state = app.invoke(initial_state)
        results.append({
            "subject": subject,
            "category": final_state.get("category") or "N/A",
            "priority": final_state.get("priority") or "N/A",
            "guard_verdict": final_state.get("guard_verdict") or "N/A",
            "final_route": final_state.get("route") or "N/A",
            "draft_reply": final_state.get("draft_reply") or "",
        })

    # Print clean summary table
    col_subj = 38
    col_cat = 25
    col_prio = 10
    col_guard = 15
    col_route = 16
    col_draft = 32

    sep_line = (
        f"+{'-' * col_subj}"
        f"+{'-' * col_cat}"
        f"+{'-' * col_prio}"
        f"+{'-' * col_guard}"
        f"+{'-' * col_route}"
        f"+{'-' * col_draft}+"
    )

    header = (
        f"| {'Email Subject':<{col_subj-2}} "
        f"| {'Category':<{col_cat-2}} "
        f"| {'Priority':<{col_prio-2}} "
        f"| {'Guard Verdict':<{col_guard-2}} "
        f"| {'Final Route':<{col_route-2}} "
        f"| {'Draft Reply (first 100)':<{col_draft-2}} |"
    )

    print("\n" + "=" * 140)
    print("FINAL PIPELINE EXECUTION SUMMARY TABLE")
    print("=" * 140)
    print(sep_line)
    print(header)
    print(sep_line)

    for r in results:
        subj = (r["subject"][: col_subj - 5] + "...") if len(r["subject"]) > col_subj - 2 else r["subject"]
        cat = r["category"][: col_cat - 2]
        prio = r["priority"][: col_prio - 2]
        guard = r["guard_verdict"][: col_guard - 2]
        route = r["final_route"][: col_route - 2]

        draft_clean = r["draft_reply"].replace("\n", " ").strip()
        first_100 = (draft_clean[: col_draft - 5] + "...") if len(draft_clean) > col_draft - 2 else draft_clean
        if not first_100:
            first_100 = "[No draft - Routed directly]"

        row = (
            f"| {subj:<{col_subj-2}} "
            f"| {cat:<{col_cat-2}} "
            f"| {prio:<{col_prio-2}} "
            f"| {guard:<{col_guard-2}} "
            f"| {route:<{col_route-2}} "
            f"| {first_100:<{col_draft-2}} |"
        )
        print(row)

    print(sep_line)
    print("=" * 140 + "\n")
    return results


if __name__ == "__main__":
    run_demo()
