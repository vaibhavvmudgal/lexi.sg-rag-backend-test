# ingest.py
import os
import pickle
import re
import warnings

from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadWarning
from docx import Document

# 1. CONFIG
DOCS_DIR = Path("legal_docs")
EMBEDDINGS_DIR = Path("embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 2. LOAD EMBEDDER
embedder = SentenceTransformer(MODEL_NAME)

def load_doc(path: Path) -> str:
    if path.suffix == ".pdf":
        # suppress startxref warnings and allow non-strict parsing
        warnings.filterwarnings("ignore", category=PdfReadWarning)
        reader = PdfReader(str(path), strict=False)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text on sentence boundaries so each chunk starts and ends with a full sentence.
    """
    # Split by sentence-ending punctuation (., !, ?) followed by whitespace
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # If adding this sentence exceeds the limit, finalize the current chunk
        if current and len(current) + len(sentence) > chunk_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence

    # Add any remaining text as the last chunk
    if current:
        chunks.append(current.strip())

    return chunks


def main():
    docs = list(DOCS_DIR.glob("*.pdf")) + list(DOCS_DIR.glob("*.docx"))
    all_embeddings: List[np.ndarray] = []
    metadata: List[dict] = []

    for doc_path in docs:
        text = load_doc(doc_path)
        for idx, chunk in enumerate(chunk_text(text)):
            emb = embedder.encode(chunk)
            all_embeddings.append(emb)
            metadata.append({
                "source": doc_path.name,
                "chunk_id": idx,
                "text": chunk
            })

    if not all_embeddings:
        print("No documents found or no text extracted.")
        return

    # Build FAISS index
    dim = all_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(all_embeddings).astype('float32'))

    # Persist index and metadata
    faiss.write_index(index, str(EMBEDDINGS_DIR / "faiss_index.bin"))
    with open(EMBEDDINGS_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
