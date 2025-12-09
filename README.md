# Building Production-Ready Agentic RAG with LangGraph and AWS Bedrock AgentCore

## Overview

Retrieval-Augmented Generation (RAG) is a foundational pattern for grounding LLM responses in factual knowledge. However, **traditional RAG pipelines are brittle**: they retrieve once, generate once, and implicitly assume the retrieved context is sufficient. In real-world scenarios, this assumption frequently breaks, resulting in incomplete answers, hallucinations, and inconsistent quality.

This repository implements a **production-ready Agentic RAG system** that introduces reasoning, validation, and control flow into the retrieval stage. Instead of blindly generating answers, the system evaluates the quality of retrieved evidence, decides whether to retrieve again, and only proceeds to generation when it is confident that sufficient context exists.

The system combines:
- **LangGraph** for explicit agent orchestration
- **AWS Bedrock AgentCore** for managed runtime, memory, and observability
- **Pinecone** for vector storage and retrieval
- **RAGAS** for objective evaluation of retrieval and answer quality

The result is a **plan–act–check Agentic RAG pipeline** suitable for real production workloads.

---

## Why Traditional RAG Falls Short

Traditional RAG follows a linear design:

1. Embed the query  
2. Retrieve top-k chunks  
3. Generate an answer  

### Key Limitations
- **Single-shot retrieval**  
  Weak or incomplete context still leads to generation.
- **No retrieval validation**  
  There is no metric-based check for relevance or sufficiency.
- **No iterative refinement**  
  The system cannot adjust queries or retry retrieval.
- **Hallucination-prone**  
  Models answer confidently even when evidence is thin.
- **Hard to audit in production**  
  Retrieval failures are opaque and hard to diagnose.

Traditional RAG *hopes* retrieval works.  
Agentic RAG **verifies** that it works.

---

## Why Agentic RAG

Agentic RAG introduces decision-making into retrieval:

- Retrieval becomes **iterative**, not one-shot
- Evidence is **explicitly evaluated**
- Generation is **gated** on sufficiency
- Failures are **intentional and safe**

This shifts RAG from a passive pattern to an **evidence-driven reasoning system**.

---

## High-Level Architecture

**Flow**

User Query  
→ Retrieval + Grader Agent  
→ (iterative loop if needed)  
→ Generator Agent  
→ Final Answer / Refusal

**Core Principles**
- Evidential sufficiency before generation
- Objective evaluation using RAGAS
- Bounded retrieval loops
- Full observability and auditability

---

## Data Preparation & Pinecone Indexing

### Source Documents
- *Attention Is All You Need*
- *BERT*
- *CLIP*

### Processing Pipeline
- PDF parsing via **Docling**
- Text cleaning and normalization
- Standard metadata (paper id, source, page, section)
- Semantic chunking with tuned size/overlap
- 1024-dimensional embeddings

### Pinecone Index Design
- Metric: Cosine
- Region: us-east-1
- Namespaces:
  - `default` → document chunks for retrieval
  - `ground-truth` → curated Q&A exemplars

### Indexed Records
| Namespace | Purpose | Vectors |
|--------|--------|--------|
| default | Retrieval corpus | 915 |
| ground-truth | Golden Q&A dataset | 134 |
| **Total** | | **1,049** |

This dual-namespace design enables **both retrieval and objective validation**.

---

## Tools Used by the Agent

### `retrieve_documents`
Retrieves top-k chunks from the research-paper corpus with lightweight metadata. Used iteratively during evidence gathering.

### `retrieve_ground_truth`
Fetches semantically similar Q&A exemplars from the curated golden dataset, including:
- Expected answer
- Source paper
- `min_chunks_required` hint

### `evaluate_retrieval_quality`
Evaluates retrieved documents using **RAGAS**:
- Context Precision
- Context Recall  
Returns a sufficiency verdict that gates generation.

### `evaluate_answer_quality` (Prepared)
Designed to evaluate final answers using:
- Faithfulness
- Answer Relevancy  
Prepared for future integration as a final safety gate.

---

## Retrieval + Grader Agent

The Retrieval+Grader Agent is the **evidence gatekeeper** of the system.

### Responsibilities
- Retrieve relevant context
- Validate evidence using RAGAS
- Decide whether to retry or stop

### Capabilities
1. `retrieve_documents`
2. `retrieve_ground_truth`
3. `evaluate_retrieval_quality`

### Behavior
- Runs in bounded attempts
- Maintains agent state:
  - Retrieved context
  - Evaluation results
  - Sufficiency flag
- Emits a binary verdict:
  - **SUFFICIENT**
  - **INSUFFICIENT**

This transforms retrieval into a **controlled plan–act–check loop**.

---

## `should_continue_retrieval` Router

Controls retrieval looping logic.

### Decision Rules
- Stop immediately if `is_sufficient == True`
- Retry while retrieval budget remains
- Enforce a hard cap to prevent infinite loops

If budget is exhausted, the system exits cleanly and follows the failure policy (refusal or clarification).

---

## Generator Agent

The Generator Agent produces the final answer **only when evidence is sufficient**.

### Behavior
- Checks `is_sufficient`
- If false → returns a refusal or clarification message
- If true:
  - Builds a strict context block
  - Enforces “answer only from context” behavior
  - Uses `gpt-4o-mini` with temperature = 0.0
  - Performs a single deterministic generation

This minimizes variance and prevents hallucinations.

---

## Why AWS Bedrock AgentCore

AWS Bedrock AgentCore provides the production runtime foundation.

### Problems It Solves
- Session management
- Short- and long-term memory
- IAM-based security
- Deployment standardization
- Observability and tracing

### Benefits
- CloudWatch logs for every tool call
- X-Ray traces for latency and failures
- Managed ARM64 runtime
- No custom state store or server layer

This allows focus on **Agentic RAG quality**, not infrastructure wiring.

---

## Pre-AgentCore Setup

- IAM user with:
  - AdministratorAccess
  - Bedrock AgentCore Full Access
- AWS CLI configured with credentials
- Clean entrypoint decorated with `@app.entrypoint`
- Dependencies defined in `requirements.txt`

---

## AgentCore Configuration

```bash
agentcore configure -e Agent/main.py
