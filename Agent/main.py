import logging
import warnings
from langgraph.graph import StateGraph, END

from state import AgentState
from retrieval_grader_agent import retrieval_grader_agent
from generator_agent import generator_agent
from router import should_continue_retrieval

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Optional: quiet logs
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

# Create the AgentCore app instance
app = BedrockAgentCoreApp()


def create_workflow():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("retrieve_and_grade", retrieval_grader_agent)
    graph.add_node("generate", generator_agent)

    # Entry point
    graph.set_entry_point("retrieve_and_grade")

    # Conditional routing after retrieval
    graph.add_conditional_edges(
        "retrieve_and_grade",
        should_continue_retrieval,
        {
            "retry": "retrieve_and_grade",
            "stop": "generate",
        },
    )

    # Generator always ends the workflow
    graph.add_edge("generate", END)

    return graph.compile()


# Create workflow once (outside entrypoint for reuse)
workflow = create_workflow()


# AgentCore Entrypoint
@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation in AgentCore runtime"""
    print("Received payload:", payload)
    print("Context:", context)
    
    # Extract question from payload
    question = payload.get("prompt", "")
    
    # Invoke the workflow
    result = workflow.invoke({
        "question": question,
        "messages": [],
        "retrieval_count": 0,
        "is_sufficient": False,
        "retrieved_docs_json": "",
        "final_answer": "",
    })
    
    #print("Result:", result)
    
    # Return the answer in the expected format
    return {
        "result": result["final_answer"],
        "metadata": {
            "is_sufficient": result["is_sufficient"],
            "retrieval_count": result["retrieval_count"]
        }
    }


if __name__ == "__main__":
    app.run()