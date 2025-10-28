# ⚡ Quick Start Guide

## 🏃 1-Minute Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI key (or skip to use fallback model)
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run the server
python app.py
```

✅ Server running at: `http://localhost:8000`

## 🧪 Test Immediately

```bash
# Test 1: Health check
curl http://localhost:8000/health

# Test 2: Ask a medical question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of myocardial infarction?", "top_k": 3}'
```

## 🚀 Deploy to Railway (2 minutes)

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login & Deploy
railway login
railway init
railway up

# 3. Set your API key
railway variables set OPENAI_API_KEY=your-key-here

# 4. Get your public URL
railway open
```

Your API is now live! 🎉

## 📋 Test Commands (Copy & Paste)

### Health Check
```bash
curl https://your-app.up.railway.app/health
```

### Cardiology Question
```bash
curl -X POST https://your-app.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the signs of acute coronary syndrome?",
    "top_k": 5
  }'
```

### Pharmacology Question
```bash
curl -X POST https://your-app.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the contraindications of penicillin?",
    "top_k": 3
  }'
```

### Emergency Medicine
```bash
curl -X POST https://your-app.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do you manage anaphylactic shock?",
    "top_k": 4
  }'
```

## 📊 Expected Response

```json
{
  "answer": "Short, medically accurate answer based on retrieved contexts...",
  "contexts": [
    "Relevant context snippet 1 from medical textbooks...",
    "Relevant context snippet 2...",
    "Relevant context snippet 3..."
  ]
}
```

## 🔥 Common Issues

| Problem | Solution |
|---------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| "FAISS not found" | Ensure `outputs/` folder exists |
| "OpenAI error" | App auto-fallbacks to open-source model |
| Slow responses | Reduce `top_k` to 3 |

## 📚 Documentation

- **Full Deployment Guide**: See `DEPLOYMENT.md`
- **API Documentation**: See `API_README.md`
- **Original Requirements**: See `README.md`

## ✅ Pre-Submission Checklist

- [ ] API deployed and publicly accessible
- [ ] Tested with curl or Postman
- [ ] Responses < 60 seconds
- [ ] Returns HTTP 200
- [ ] JSON format: `{"answer": "...", "contexts": [...]}`
- [ ] Answers are medically accurate
- [ ] Contexts are relevant

## 🎯 Key Parameters

- **top_k**: 3-5 (recommended)
- **Response time**: < 60 seconds (guaranteed)
- **Status code**: 200 (always)
- **Max query length**: No hard limit
- **Context length**: Truncated to 300 chars each

---

Need help? Check `DEPLOYMENT.md` for detailed troubleshooting!

