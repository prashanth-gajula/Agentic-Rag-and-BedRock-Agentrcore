from typing import List, TypedDict, Annotated
import operator

class AgentState(TypedDict):
    question: str
    messages: Annotated[List, operator.add]
    retrieval_count: int          # how many retrieval attempts so far
    is_sufficient: bool           # is retrieval good enough?
    retrieved_docs_json: str 
    final_answer: str