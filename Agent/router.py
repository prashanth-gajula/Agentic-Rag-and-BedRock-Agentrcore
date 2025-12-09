from state import AgentState


def should_continue_retrieval(state: AgentState) -> str:
    """
    Decide whether to:
    - retry retrieval (if not sufficient and attempts < 2)
    - or stop (END) after success or after 2 attempts
    """
    is_sufficient = state.get("is_sufficient", False)
    retrieval_count = state.get("retrieval_count", 0)

    if is_sufficient:
        print(f"\n  ✅ ROUTING: Documents sufficient → END")
        return "stop"

    if retrieval_count >= 2:
        # We already tried 2 times; don't keep looping forever
        print(f"\n  ⚠️ ROUTING: Max attempts reached (2) → END")
        return "stop"

    # Otherwise, try retrieval again
    print(f"\n  🔄 ROUTING: Insufficient → Retry Retrieval ({retrieval_count}/2 so far)")
    return "retry"