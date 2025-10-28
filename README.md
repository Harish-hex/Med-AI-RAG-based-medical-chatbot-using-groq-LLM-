# 🧠 PRD: Medical RAG Chatbot API (Production-Ready, Evaluation Compliant)

## 🎯 Objective
Build a **publicly accessible REST API** for a **Medical Retrieval-Augmented Generation (RAG)** chatbot that answers medical questions using citation-backed, explainable context retrieved from prebuilt vector embeddings.

The system must:
- Accept a JSON POST request with `{ query, top_k }`
- Retrieve relevant context snippets from FAISS
- Generate a concise and correct answer using an LLM
- Return JSON `{ answer, contexts }` within **60 seconds**

---

## 🧮 Evaluation Metrics & Scoring

| Metric | Weight | Description |
|---------|--------|--------------|
| **Answer Relevancy** | 30% | The answer directly addresses the question. |
| **Answer Correctness** | 30% | The answer is factually accurate. |
| **Context Relevance** | 25% | Retrieved context closely relates to the question. |
| **Faithfulness** | 15% | The answer is consistent with provided context. |

### 💡 Tips to Maximize Score
- Keep answers **short, direct, and correct**.
- Retrieve **3–5** context snippets using `top_k`.
- Avoid hallucinations — if unsure, respond conservatively.
- Prioritize medically accurate and verifiable phrasing.

---

## ⚙️ Endpoint Requirements

| Property | Description |
|-----------|-------------|
| **Route** | `POST /ask` |
| **Content-Type** | `application/json` |
| **Timeout** | Must respond within **60 seconds** |
| **Status Code** | `200` on success |
| **Public Access** | Endpoint must be reachable via URL |
| **Empty Contexts** | Allowed, but may affect context score |

### ✅ Request (JSON)
```json
{
  "query": "user question",
  "top_k": 5
}