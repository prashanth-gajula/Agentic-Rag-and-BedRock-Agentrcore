from typing import List
import json
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from prompts import retriver_grader_prompt
import config as config
from tools import ALL_TOOLS
from state import AgentState


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=config.OPENAI_API_KEY,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


def retrieval_grader_agent(state: AgentState) -> AgentState:
    """
    Single agent that:
      - Retrieves docs (RAG)
      - Retrieves ground truth
      - Evaluates retrieval quality
      - Decides if retrieval is SUFFICIENT or INSUFFICIENT

    This function represents ONE retrieval attempt.
    The graph (routing logic) will call it up to 3 times.
    """
    question = state["question"]
    retrieval_count = state.get("retrieval_count", 0)
    messages: List = state.get("messages", [])

    attempt_num = retrieval_count + 1
    print("\n" + "=" * 80)
    print(f"🤖 RETRIEVAL+GRADER AGENT (Attempt {attempt_num})")
    print("=" * 80 + "\n")

    system_prompt = retriver_grader_prompt

    # For this attempt, we append a fresh System + Human message
    messages = list(messages)  # copy to avoid side effects
    messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=f"Question: {question}"))

    # Values we will update while tools run
    retrieved_docs_json = state.get("retrieved_docs_json", "")
    is_sufficient = False

    # Tool execution node
    tool_node = ToolNode(ALL_TOOLS)

    # Let the LLM iterate between thinking and tools
    max_tool_iterations = 3
    for _ in range(max_tool_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # If the LLM is calling tools
        if getattr(response, "tool_calls", None):
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(f"  🔧 LLM calling tools: {', '.join(tool_names)}")

            # Execute tools and add their outputs
            tool_results = tool_node.invoke({"messages": [response]})

            for tool_msg in tool_results["messages"]:
                messages.append(tool_msg)

                if isinstance(tool_msg, ToolMessage):
                    # We can check which tool produced this output
                    tool_name = getattr(tool_msg, "name", None)
                    content_str = tool_msg.content

                    # Store latest retrieved docs
                    if tool_name == "retrieve_documents":
                        retrieved_docs_json = content_str
                        try:
                            parsed = json.loads(content_str)
                            print(f"     → Retrieved {parsed.get('retrieved_count', 0)} documents")
                        except Exception:
                            print("     → Retrieved documents (unable to parse JSON for logging)")

                    # Log retrieval evaluation
                    if tool_name == "evaluate_retrieval_quality":
                        try:
                            eval_data = json.loads(content_str)
                            cp = eval_data.get("context_precision")
                            cr = eval_data.get("context_recall")
                            is_sufficient = eval_data.get("is_sufficient", False)

                            print("\n  📊 Retrieval Evaluation:")
                            print(f"     • Context Precision: {cp}")
                            print(f"     • Context Recall:    {cr}")
                            print(f"     • Sufficient:        {is_sufficient}")
                        except Exception:
                            print("     → Error parsing evaluation JSON")

        else:
            # No more tool calls → LLM is giving its decision
            decision_text = response.content.strip().upper()
            print(f"\n  💭 LLM Decision this attempt: {decision_text}")

            if "SUFFICIENT" in decision_text and "INSUFFICIENT" not in decision_text:
                is_sufficient = True
            elif "INSUFFICIENT" in decision_text:
                is_sufficient = False
            # If it's something else, we just keep current is_sufficient

            break  # end this attempt loop

    # Update and return state
    new_state: AgentState = {
        **state,
        "messages": messages,
        "retrieval_count": attempt_num,
        "is_sufficient": is_sufficient,
        "retrieved_docs_json": retrieved_docs_json,
    }

    print(f"\n  ✅ Attempt {attempt_num} complete → is_sufficient={is_sufficient}")
    return new_state