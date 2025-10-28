# 🏥 Medical RAG Chatbot API

A production-ready **Retrieval-Augmented Generation (RAG)** API for answering medical questions with citation-backed context from medical textbooks.

## 📊 System Overview

```
User → POST /ask → Embed Query → FAISS Search → LLM Generation → Response
```

**Key Features:**
- ✅ Fast semantic search using FAISS (35,167 medical text chunks)
- ✅ OpenAI GPT-4-mini integration with automatic fallback
- ✅ Citation-backed answers with source contexts
- ✅ Sub-60 second response times
- ✅ Production-ready with error handling
- ✅ Railway/Render deployment ready

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Your API Key

```bash
# Copy the example file
cp env.example .env

# Edit .env and add your OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 3. Run the Server

```bash
python app.py
```

Server starts at: `http://localhost:8000`

### 4. Test It

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of myocardial infarction?",
    "top_k": 5
  }'
```

## 📡 API Endpoints

### `GET /`
Health check and API information

**Response:**
```json
{
  "status": "healthy",
  "service": "Medical RAG Chatbot API",
  "version": "1.0.0"
}
```

### `GET /health`
Detailed health status

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "faiss_loaded": true,
  "metadata_loaded": true,
  "llm_type": "openai"
}
```

### `POST /ask` ⭐
Main endpoint for medical questions

**Request:**
```json
{
  "query": "What are the contraindications of aspirin?",
  "top_k": 3
}
```

**Parameters:**
- `query` (string, required): The medical question
- `top_k` (integer, optional): Number of context snippets to retrieve (default: 3, max: 10)

**Response:**
```json
{
  "answer": "Aspirin is contraindicated in patients with active peptic ulcer disease, bleeding disorders, severe liver disease, and in children with viral infections due to Reye's syndrome risk. It should be avoided in patients with aspirin allergy or asthma triggered by NSAIDs.",
  "contexts": [
    "Context snippet 1 from medical textbook...",
    "Context snippet 2 about contraindications...",
    "Context snippet 3 with clinical guidelines..."
  ]
}
```

**Status Codes:**
- `200`: Success (always returned, even on errors)
- Response time: < 60 seconds

## 🧪 Testing Examples

### Example 1: Cardiology
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the signs of acute coronary syndrome?",
    "top_k": 5
  }'
```

### Example 2: Pharmacology
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When should beta-blockers be avoided?",
    "top_k": 3
  }'
```

### Example 3: Emergency Medicine
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do you manage anaphylactic shock?",
    "top_k": 4
  }'
```

## 🏗️ Architecture

### Technology Stack
- **Framework**: FastAPI (async, high-performance)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS (35,167 chunks from 9 medical textbooks)
- **LLM**: OpenAI GPT-4-mini (with open-source fallback)
- **Deployment**: Railway/Render compatible

### Data Sources
- 9 medical textbooks (1.6GB PDFs)
- 18,663 total pages processed
- 35,167 text chunks with embeddings
- Topics: Cardiology, Emergency Medicine, Internal Medicine, Nephrology, Gastrology, Anatomy, Dentistry, Infectious Disease

### Response Flow
1. **Query Embedding**: User query → SentenceTransformer → 384-dim vector
2. **Vector Search**: FAISS retrieves top-k similar chunks (L2 distance)
3. **Context Ranking**: Most relevant contexts selected
4. **LLM Generation**: GPT-4-mini generates answer using only provided contexts
5. **Response**: JSON with answer + source contexts

## 📊 Evaluation Metrics

The system is optimized for:

| Metric | Weight | Strategy |
|--------|--------|----------|
| **Answer Relevancy** | 30% | Direct, focused answers |
| **Answer Correctness** | 30% | Medically accurate, verified |
| **Context Relevance** | 25% | FAISS semantic search |
| **Faithfulness** | 15% | Strict prompt engineering |

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, fallback available)
- `PORT`: Server port (default: 8000, Railway sets automatically)

### Model Settings

In `app.py`:
```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_FILE = "outputs/medical_faiss.index"
METADATA_FILE = "outputs/medical_metadata.json"
```

### LLM Prompt Template

```python
You are a reliable medical assistant AI.
Use ONLY the following context to answer the user's question.
If the information is missing, respond: 
"I don't have enough reliable information to answer confidently."

Keep the answer concise, medically correct, and relevant.
```

## 🚢 Deployment

### Railway (Recommended)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and initialize
railway login
railway init

# Set API key
railway variables set OPENAI_API_KEY=your-key-here

# Deploy
railway up
```

Your app will be live at: `https://your-app.up.railway.app`

### Render

1. Connect GitHub repository
2. Select "Web Service"
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `OPENAI_API_KEY`

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📁 Project Structure

```
medical_chatbot_pipeline/
├── app.py                          # FastAPI server ⭐
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway/Heroku config
├── runtime.txt                     # Python version
├── env.example                     # Environment template
├── test_api.py                     # API test script
├── DEPLOYMENT.md                   # Detailed deployment guide
├── outputs/
│   ├── medical_faiss.index         # Vector database (35K chunks)
│   ├── medical_metadata.json       # Source metadata
│   ├── processing_summary.csv      # Processing stats
│   └── cleaned_texts/              # Extracted text files
└── medical_pdf_pipeline_fast.py    # PDF processing pipeline
```

## 🧪 Automated Testing

Run the test suite:

```bash
# Start server in one terminal
python app.py

# Run tests in another terminal
python test_api.py
```

Or use pytest:
```bash
pip install pytest requests
pytest test_api.py
```

## 🔍 Monitoring & Logs

### Local Development
```bash
# Verbose logging
uvicorn app:app --log-level debug
```

### Production (Railway)
- View logs in Railway dashboard
- Monitor response times
- Track error rates

## ⚠️ Troubleshooting

### Issue: "FAISS index not found"
**Solution:** Ensure `outputs/` directory exists with index files
```bash
ls outputs/
# Should show: medical_faiss.index, medical_metadata.json
```

### Issue: Slow responses
**Solution:** 
- Reduce `top_k` to 3
- Use OpenAI instead of fallback model
- Check server resources

### Issue: "OpenAI API error"
**Solution:** App automatically falls back to open-source model (Flan-T5)

### Issue: Out of memory
**Solution:** 
- Deploy on Railway/Render with sufficient RAM (2GB+ recommended)
- Reduce batch sizes if needed

## 📈 Performance

- **Average response time**: 2-5 seconds (OpenAI), 10-20 seconds (fallback)
- **Max response time**: < 60 seconds (guaranteed)
- **Vector search time**: < 100ms
- **Context retrieval**: < 200ms
- **LLM generation**: 1-4 seconds

## 🔐 Security

- API keys stored in environment variables
- CORS enabled for public access
- Input validation on all endpoints
- Rate limiting (configure as needed)
- No sensitive data logging

## 📚 Medical Knowledge Base

**Textbooks Processed:**
1. Anatomy & Physiology (1,300 pages)
2. Cardiology (2,034 pages)
3. Dentistry (710 pages)
4. Emergency Medicine (2,727 pages)
5. Gastrology (2,724 pages)
6. General Medicine (1,428 pages)
7. Infectious Disease (622 pages)
8. Internal Medicine (4,171 pages)
9. Nephrology (2,947 pages)

**Total:** 18,663 pages → 35,167 chunks with embeddings

## 🎯 Best Practices

1. **Use appropriate top_k**: 3-5 for best balance
2. **Monitor response times**: Stay well under 60s
3. **Test medical accuracy**: Verify answers with domain experts
4. **Handle errors gracefully**: Always return JSON
5. **Log important events**: Track queries and errors
6. **Update knowledge base**: Re-process PDFs as needed

## 📞 Support & Contact

For issues or questions:
1. Check `DEPLOYMENT.md` for detailed guides
2. Review logs for error messages
3. Test locally before deploying
4. Verify environment variables

## 📄 License

This project is for educational and evaluation purposes.

---

Built with ❤️ for medical education and research

