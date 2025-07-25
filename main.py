# main.py
import os
import json
import pickle
import re

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # v1 client for OpenRouter
from dotenv import load_dotenv
load_dotenv()  # reads .env in cwd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. OpenRouter / OpenAI v1 setup
apikey = os.getenv("OPEN_ROUTER")
if not apikey:
    raise RuntimeError("Please set the OPENROUTER_API_KEY environment variable")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=apikey,
)

MODEL_NAME = "qwen/qwen3-235b-a22b-07-25:free"


def generate_completion(prompt: str) -> str:
    """
    Call OpenRouter ChatCompletion API with the given prompt.
    Optional ranking headers can be passed via extra_headers.
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a legal research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


# 2. Load FAISS index & metadata
EMB_DIR = "embeddings"
index = faiss.read_index(f"{EMB_DIR}/faiss_index.bin")
with open(f"{EMB_DIR}/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3. FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    text: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]

@app.get("/")
async def root():
    return {"status": "Legal RAG backend up!"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # 3.a Embed & retriev e top-3 snippets
    q_emb = embedder.encode(req.query).astype("float32")
    D, I = index.search(np.array([q_emb]), k=2)

    # 3.b Collect snippets
    retrieved = [metadata[i] for i in I[0]]
    snippets = []
    for idx, m in enumerate(retrieved, 1):
        txt = m["text"]
        if len(txt) > 300:
            txt = txt[:300] + "..."
        snippets.append(f"[{idx}] {txt}")
    snippet_block = "\n\n".join(snippets)

    # 4. JSON-output prompt
    prompt = (
        """You are a legal research assistant. Below are numbered excerpts from case documents; each excerpt has its full text and source.  Using **only** these excerpts, answer the user’s question in one clear paragraph. If required Start your answer with “No,” or “Yes,” otherwise don't , and avoid any extra commentary. """
        f"Excerpts:\n{snippet_block}\n\n"
        f"Question: {req.query}\n"
        "JSON:"
    )

    # 5. Call the LLM
    raw = generate_completion(prompt)

    # 6. Parse or fallback
    try:
        data = json.loads(raw)
        if not ("answer" in data and "citations" in data):
            raise ValueError("Missing keys")
        return data
    except Exception:
        return {
            "answer": raw,
            "citations": [
                {"text": m["text"], "source": m["source"]}
                for m in retrieved
            ],
        }
