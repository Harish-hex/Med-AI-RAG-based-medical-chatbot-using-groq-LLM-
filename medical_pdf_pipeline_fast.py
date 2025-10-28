import os
import re
import json
import fitz  # PyMuPDF - much faster than pdfplumber
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------- CONFIG ----------
PDF_DIR = "medical_books/"             # folder containing your PDFs
OUTPUT_DIR = "outputs/"                # folder for all outputs
CHUNK_SIZE = 800                       # tokens (approx)
CHUNK_OVERLAP = 200
VECTOR_DB_FILE = "outputs/medical_faiss.index"
METADATA_FILE = "outputs/medical_metadata.json"
CLEANED_TEXT_DIR = "outputs/cleaned_texts/"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_PDFS = None                        # Limit number of PDFs to process (set to None for all)
MAX_PAGES_PER_PDF = None               # Limit pages per PDF (set to None for all pages)
PARALLEL_WORKERS = min(4, cpu_count()) # Number of parallel workers
SKIP_EMBEDDINGS = False                # Set to True to skip embedding generation (faster)

# ---------- 1️⃣ PDF Extraction with PyMuPDF (FAST) ----------
def extract_text_from_pdf(pdf_path):
    """Extract text using PyMuPDF - much faster than pdfplumber"""
    text_pages = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = total_pages if MAX_PAGES_PER_PDF is None else min(MAX_PAGES_PER_PDF, total_pages)
        
        for i in range(pages_to_process):
            try:
                page = doc[i]
                text = page.get_text()
                
                # Clean text: normalize whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                if text:
                    text_pages.append({"page": i+1, "text": text})
            except Exception as e:
                print(f"⚠️ Skipping page {i+1} in {pdf_path}: {e}")
                continue
        
        doc.close()
        
        if MAX_PAGES_PER_PDF and total_pages > MAX_PAGES_PER_PDF:
            return text_pages, f"Processed {pages_to_process}/{total_pages} pages"
        
        return text_pages, None
    
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return [], None

# ---------- 2️⃣ Text Chunking ----------
def chunk_text(text, chunk_size=800, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---------- 3️⃣ Save Cleaned Text ----------
def save_cleaned_text(book_name, pages):
    """Save cleaned text from each PDF to a separate file"""
    os.makedirs(CLEANED_TEXT_DIR, exist_ok=True)
    output_file = os.path.join(CLEANED_TEXT_DIR, f"{book_name}_cleaned.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Cleaned Text from: {book_name}\n")
        f.write(f"Total Pages: {len(pages)}\n")
        f.write(f"{'='*60}\n\n")
        
        for page_data in pages:
            f.write(f"\n--- Page {page_data['page']} ---\n")
            f.write(page_data['text'])
            f.write("\n\n")

# ---------- 4️⃣ Process Single PDF (for parallel processing) ----------
def process_single_pdf(pdf_file):
    """Process a single PDF - used for parallel processing"""
    book_name = os.path.splitext(pdf_file)[0]
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    
    pages, note = extract_text_from_pdf(pdf_path)
    
    if not pages:
        return {"book": book_name, "chunks": [], "pages": 0, "note": "No text extracted"}
    
    # Save cleaned text
    save_cleaned_text(book_name, pages)
    
    # Create chunks
    all_chunks = []
    for page_data in pages:
        for chunk in chunk_text(page_data["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append({
                "book": book_name,
                "page": page_data["page"],
                "text": chunk
            })
    
    return {
        "book": book_name,
        "chunks": all_chunks,
        "pages": len(pages),
        "note": note
    }

# ---------- 5️⃣ Build Chunks from All PDFs (with parallel processing) ----------
def build_chunks():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{PDF_DIR}'")
    
    # Limit number of PDFs if MAX_PDFS is set
    if MAX_PDFS is not None:
        pdf_files = pdf_files[:MAX_PDFS]
        print(f"📌 Processing first {len(pdf_files)} PDF(s)")
    
    if MAX_PAGES_PER_PDF:
        print(f"📌 Limiting to first {MAX_PAGES_PER_PDF} pages per PDF")
    
    print(f"🚀 Using {PARALLEL_WORKERS} parallel workers\n")
    
    # Process PDFs in parallel
    all_chunks = []
    with Pool(PARALLEL_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_pdf, pdf_files),
            total=len(pdf_files),
            desc="Processing PDFs"
        ))
    
    # Combine results
    for result in results:
        all_chunks.extend(result["chunks"])
        print(f"   ✓ {result['book']}: {result['pages']} pages, {len(result['chunks'])} chunks" + 
              (f" ({result['note']})" if result['note'] else ""))
    
    return pd.DataFrame(all_chunks)

# ---------- 6️⃣ Embedding + Vector DB ----------
def create_vector_db(df):
    if SKIP_EMBEDDINGS:
        print("\n⏭️  Skipping embedding generation (SKIP_EMBEDDINGS=True)")
        print(f"✅ Extracted and cleaned {len(df)} chunks from PDFs")
        return
    
    print("\n🔹 Generating embeddings...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32  # Adjust based on your GPU/CPU
    )

    print("🔹 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, VECTOR_DB_FILE)

    metadata = df.to_dict(orient="records")
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save summary CSV
    summary_file = os.path.join(OUTPUT_DIR, "processing_summary.csv")
    summary = df.groupby("book").agg(
        total_chunks=("text", "count"),
        pages=("page", "max")
    ).reset_index()
    summary.to_csv(summary_file, index=False)

    print(f"\n✅ Done! Indexed {len(df)} chunks.")
    print(f"📁 Vector DB: {VECTOR_DB_FILE}")
    print(f"📁 Metadata : {METADATA_FILE}")
    print(f"📁 Summary  : {summary_file}")

# ---------- 7️⃣ Quick Search Function ----------
def search(query, top_k=5):
    if SKIP_EMBEDDINGS or not os.path.exists(VECTOR_DB_FILE):
        print("❌ No vector database found. Run with SKIP_EMBEDDINGS=False first.")
        return
    
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(VECTOR_DB_FILE)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)

    print("\n🔍 Top Results:")
    for rank, idx in enumerate(I[0]):
        data = metadata[idx]
        snippet = data["text"][:250].replace("\n", " ")
        print(f"{rank+1}. {data['book']} (p.{data['page']}): {snippet}...\n")

# ---------- MAIN ----------
if __name__ == "__main__":
    print("="*70)
    print("🏥 FAST Medical PDF Pipeline (PyMuPDF + Parallel Processing)")
    print("="*70)
    
    df = build_chunks()
    
    if not df.empty:
        create_vector_db(df)
        
        if not SKIP_EMBEDDINGS:
            # Test a query
            print("\n" + "="*70)
            print("🧪 Testing search functionality...")
            print("="*70)
            search("What are the contraindications of penicillin in pregnancy?")
    else:
        print("❌ No chunks were created. Check your PDFs.")

