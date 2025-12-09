import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import config as config
from state import AgentState


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=config.OPENAI_API_KEY,
)


def generator_agent(state: AgentState) -> AgentState:
    """
    Simple generator:
    - If retrieval is insufficient → return fallback.
    - Else → generate answer ONCE using retrieved context.
    """
    question = state["question"]
    is_sufficient = state.get("is_sufficient", False)
    retrieved_docs_json = state.get("retrieved_docs_json", "")

    print("\n" + "=" * 80)
    print("🤖 GENERATOR AGENT")
    print("=" * 80 + "\n")

    # 1️⃣ If retrieval failed, do not generate
    if not is_sufficient:
        print("  ⚠️ Retrieval insufficient → returning fallback.")
        fallback = (
            "Sorry, I don't have enough reliable information to answer this question.\n\n"
            "Please try:\n"
            "- Rephrasing the question with more detail, or\n"
            "- Asking a different but related question."
        )
        return {
            **state,
            "final_answer": fallback,
        }

    # 2️⃣ Build context from retrieved documents
    try:
        retrieved_data = json.loads(retrieved_docs_json)
        docs = retrieved_data.get("documents", [])
    except Exception:
        docs = []

    context_blocks = []
    for doc in docs:
        source = doc.get("source", "unknown")
        content = doc.get("content", "")
        context_blocks.append(f"[{source}]\n{content}")

    context = "\n\n".join(context_blocks)

    print(f"  📄 Generating answer using {len(docs)} context documents")

    # 3️⃣ Generate answer ONCE
    prompt = f"""
You are a helpful technical assistant.

Answer the question using ONLY the context below.
If the context does not contain the answer, say you do not know.

Question:
{question}

Context:
{context}

Answer:
""".strip()

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "final_answer": response.content,
    }