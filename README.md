# Lexi.sg RAG Backend

A Retrieval-Augmented Generation (RAG) backend for legal queries with citations.

## Prerequisites
- Python 3.11  

## Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/vaibhavvmudgal/lexi.sg-rag-backend-test
    cd lexi.sg-rag-backend-test
    ```

2. **Create a virtual environment & install dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate       # macOS/Linux
    # venv\Scripts\activate       # Windows PowerShell

    pip install -r requirements.txt
    ```

3. **Configure your OpenRouter API key**

    Then edit `.env` and set:
    ```ini
    OPENROUTER_API_KEY=sk-or-<your_openrouter_api_key>
    ```
  

4. **Build embeddings & FAISS index**

    Place all your legal `.pdf` and `.docx` files into `legal_docs/`, then run:
    ```bash
    python ingest.py
    ```
    This generates:
    - `embeddings/faiss_index.bin`
    - `embeddings/metadata.pkl`

5. **Run the FastAPI server**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## Usage

### Interactive API docs
Open in your browser:

http://127.0.0.1:8000/docs
or
http://57.183.12.62:8080/docs (Deployed here)

### Use the **POST /query** endpoint to submit legal question(s).
Eg: ```{
        "query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
        }```



