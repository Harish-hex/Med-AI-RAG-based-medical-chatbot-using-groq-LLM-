import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Chatbot API",
    description="Retrieval-Augmented Generation API for medical questions",
    version="1.0.0"
)

# Add CORS middleware for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
embedding_model = None
reranker_model = None  # Cross-encoder for reranking
faiss_index = None
metadata = None
llm_client = None
use_openai = False

# Configuration
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_FILE = "outputs/medical_faiss.index"
METADATA_FILE = "outputs/medical_metadata.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def initialize_models():
    """Initialize all models and load data on startup"""
    global embedding_model, reranker_model, faiss_index, metadata, llm_client, use_openai
    
    try:
        # Load sentence transformer model
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        embedding_model = SentenceTransformer(EMBED_MODEL)
        logger.info("✅ Embedding model loaded successfully")
        
        # Load cross-encoder for reranking (improves context relevance)
        logger.info("Loading cross-encoder for reranking...")
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("✅ Cross-encoder loaded successfully")
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from: {VECTOR_DB_FILE}")
        if not os.path.exists(VECTOR_DB_FILE):
            raise FileNotFoundError(f"FAISS index not found at {VECTOR_DB_FILE}")
        faiss_index = faiss.read_index(VECTOR_DB_FILE)
        logger.info(f"✅ FAISS index loaded: {faiss_index.ntotal} vectors")
        
        # Load metadata
        logger.info(f"Loading metadata from: {METADATA_FILE}")
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Metadata file not found at {METADATA_FILE}")
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        logger.info(f"✅ Metadata loaded: {len(metadata)} entries")
        
        # Initialize LLM (Groq, OpenAI, or fallback)
        if GROQ_API_KEY and GROQ_API_KEY.strip():
            try:
                from openai import OpenAI
                llm_client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1"
                )
                # Test the API key
                llm_client.models.list()
                use_openai = True
                logger.info("✅ Groq API initialized successfully (FREE & FAST!)")
            except Exception as e:
                logger.warning(f"⚠️ Groq initialization failed: {e}")
                logger.info("📦 Falling back to open-source model")
                use_openai = False
                initialize_fallback_llm()
        elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
            try:
                from openai import OpenAI
                llm_client = OpenAI(api_key=OPENAI_API_KEY)
                # Test the API key with a minimal request
                llm_client.models.list()
                use_openai = True
                logger.info("✅ OpenAI API initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI initialization failed: {e}")
                logger.info("📦 Falling back to open-source model")
                use_openai = False
                initialize_fallback_llm()
        else:
            logger.info("📦 No API key found, using open-source model")
            use_openai = False
            initialize_fallback_llm()
            
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise

def initialize_fallback_llm():
    """Initialize fallback open-source LLM"""
    global llm_client
    try:
        from transformers import pipeline
        logger.info("Loading open-source model: google/flan-t5-base")
        llm_client = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            device=-1  # CPU
        )
        logger.info("✅ Fallback model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load fallback model: {e}")
        llm_client = None

def retrieve_context(query: str, top_k: int = 5) -> List[str]:
    """
    Two-stage retrieval with cross-encoder reranking for better context relevance.
    
    Stage 1: FAISS retrieves top-N candidates (fast, broad search)
    Stage 2: Cross-encoder reranks candidates (accurate, focused)
    
    Args:
        query: User's medical question
        top_k: Number of top matches to return
        
    Returns:
        List of relevant context strings (reranked for quality)
    """
    try:
        # Stage 1: Retrieve more candidates than needed from FAISS
        initial_k = min(top_k * 2, 15)  # Get 2x candidates for reranking
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, initial_k)
        
        # Extract candidate contexts
        candidates = []
        for idx in indices[0]:
            if idx < len(metadata):
                text = metadata[idx]["text"]
                # Use longer chunks for reranking (will be truncated after selection)
                candidates.append({
                    "text": text,
                    "idx": idx
                })
        
        if not candidates:
            return []
        
        # Stage 2: Rerank with cross-encoder for better relevance
        logger.info(f"Reranking {len(candidates)} candidates with cross-encoder...")
        
        # Create query-context pairs for reranking
        pairs = [[query, cand["text"][:1000]] for cand in candidates]  # Use first 1000 chars for scoring
        
        # Get reranking scores
        scores = reranker_model.predict(pairs)
        
        # Sort candidates by reranking score (higher is better)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        # Extract final top-k contexts (now properly reranked)
        contexts = []
        for rank_idx in ranked_indices:
            text = candidates[rank_idx]["text"]
            # Moderate truncation for final context
            context = text[:700] + "..." if len(text) > 700 else text
            contexts.append(context)
        
        logger.info(f"Retrieved and reranked {len(contexts)} context snippets")
        return contexts
        
    except Exception as e:
        logger.error(f"Error in retrieve_context: {e}")
        # Fallback to simple retrieval if reranking fails
        try:
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
            distances, indices = faiss_index.search(query_embedding, top_k)
            contexts = []
            for idx in indices[0]:
                if idx < len(metadata):
                    text = metadata[idx]["text"]
                    context = text[:700] + "..." if len(text) > 700 else text
                    contexts.append(context)
            logger.warning("Used fallback retrieval (no reranking)")
            return contexts
        except:
            return []

def generate_answer_openai(query: str, contexts: List[str]) -> str:
    """Generate answer using Groq or OpenAI API"""
    try:
        # Construct the prompt
        joined_contexts = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are a reliable medical assistant AI. Answer using ONLY the provided context.

Instructions:
- Answer the exact question asked directly and concisely
- For Yes/No questions: Start with "Yes" or "No", then explain briefly
- Use specific medical terminology when present in context
- Extract all relevant information from the context provided
- If the context truly lacks the information needed, state: "I don't have enough reliable information to answer confidently."
- NEVER add information not present in the context
- Stay faithful to the provided context at all times

Context:
{joined_contexts}

Question: {query}

Answer based ONLY on the context above:"""

        # Determine which model to use (Groq or OpenAI)
        if GROQ_API_KEY and GROQ_API_KEY.strip():
            # Use Groq's fast model (Llama 3)
            model = "llama-3.3-70b-versatile"
            logger.info("Using Groq (FREE)")
        else:
            # Use OpenAI
            model = "gpt-4o-mini"
            logger.info("Using OpenAI")
        
        # Call API
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant that provides accurate, concise answers based only on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temp for high accuracy and faithfulness
            max_tokens=350,   # Concise but complete answers
            timeout=50        # Ensure we stay within 60-second limit
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("✅ Generated answer successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error in generate_answer_openai: {e}")
        return "I encountered an error generating the answer. Please try again."

def generate_answer_fallback(query: str, contexts: List[str]) -> str:
    """Generate answer using fallback open-source model"""
    try:
        if not llm_client:
            return "I don't have enough reliable information to answer confidently."
        
        # Construct a simpler prompt for smaller models
        context_text = " ".join(contexts[:2])  # Use only first 2 contexts to stay within token limits
        prompt = f"Answer this medical question using only the given context. Question: {query} Context: {context_text}"
        
        # Generate answer
        result = llm_client(prompt, max_length=200, do_sample=False)
        answer = result[0]["generated_text"].strip()
        
        logger.info("✅ Generated answer using fallback model")
        return answer
        
    except Exception as e:
        logger.error(f"Error in generate_answer_fallback: {e}")
        return "I don't have enough reliable information to answer confidently."

def generate_answer(query: str, contexts: List[str]) -> str:
    """
    Generate answer using LLM (OpenAI or fallback).
    
    Args:
        query: User's medical question
        contexts: Retrieved context snippets
        
    Returns:
        Generated answer string
    """
    if not contexts:
        return "I don't have enough reliable information to answer confidently."
    
    if use_openai:
        return generate_answer_openai(query, contexts)
    else:
        return generate_answer_fallback(query, contexts)

@app.on_event("startup")
async def startup_event():
    """Initialize models and data when the app starts"""
    logger.info("🚀 Starting Medical RAG Chatbot API...")
    initialize_models()
    logger.info("✅ API is ready to serve requests!")

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "service": "Medical RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask - Ask a medical question"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": embedding_model is not None,
        "faiss_loaded": faiss_index is not None,
        "metadata_loaded": metadata is not None,
        "llm_type": "openai" if use_openai else "fallback"
    }

@app.post("/ask")
@app.post("/query")  # Alternative endpoint name for compatibility
async def ask_question(request: Request):
    """
    Main endpoint for asking medical questions.
    
    Request JSON:
        {
            "query": "medical question",
            "top_k": 5  (optional, default: 3)
        }
    
    Response JSON:
        {
            "answer": "concise, grounded answer",
            "contexts": ["context1", "context2", ...]
        }
    """
    try:
        # Parse request
        data = await request.json()
        query = data.get("query")
        top_k = int(data.get("top_k", 3))
        
        # Validate input
        if not query or not isinstance(query, str) or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Invalid query. Please provide a non-empty string."
            )
        
        # Validate top_k - use recommended 3-5 range for best results
        if top_k < 1 or top_k > 10:
            top_k = 5  # Default to 5 (optimal per evaluation guidelines)
        # Keep user's top_k if already in good range (3-7)
        
        logger.info(f"📝 Query received: {query[:100]}... (top_k={top_k})")
        
        # Retrieve context
        contexts = retrieve_context(query, top_k)
        
        # Generate answer
        answer = generate_answer(query, contexts)
        
        # Return response
        response = {
            "answer": answer,
            "contexts": contexts
        }
        
        logger.info(f"✅ Response generated successfully")
        return JSONResponse(status_code=200, content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        return JSONResponse(
            status_code=200,  # Return 200 even on errors as per requirements
            content={
                "answer": "I encountered an error processing your question. Please try again.",
                "contexts": []
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

