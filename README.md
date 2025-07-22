# Lexi.sg RAG Backend

A Retrieval-Augmented Generation (RAG) backend for legal queries with citations.

## Prerequisites
- Python 3.8+  
- Git  
- (Optional) GPU for faster LLM inference

## Setup

1. **Clone the repository**
    ```bash
    git clone <repo-url>
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

    Copy the example and fill in your key:
    ```bash
    cp .env.example .env
    ```
    Then edit `.env` and set:
    ```ini
    OPENROUTER_API_KEY=sk-or-<your_openrouter_api_key>
    ```
    Alternatively, export directly:
    ```bash
    export OPENROUTER_API_KEY="sk-or-..."
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

Use the **POST /query** endpoint to submit legal questions.

### cURL example
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"}'

  Example response
  {
  "answer": "No, an insurance company is not liable to pay compensation if a transport vehicle is used without a valid permit at the time of the accident. The Supreme Court held that use of a vehicle in a public place without a permit is a fundamental statutory infraction, and such a situation is not equivalent to cases involving absence of licence, fake licence, or breach of conditions such as overloading. Therefore, the insurer is entitled to recover the compensation amount from the owner and driver after paying the claim.",
  "citations": [
    {
      "text": "Use of a vehicle in a public place without a permit is a fundamental statutory infraction. The said situations cannot be equated with absence of licence or a fake licence or a licence for different kind of vehicle, or, for that matter, violation of a condition of carrying more number of passengers.",
      "source": "Doc_Name.docx"
    },
    {
      "text": "Therefore, the tribunal as well as the High Court had directed that the insurer shall be entitled to recover the same from the owner and the driver.",
      "source": "Doc_Name.docx"
    }
  ]
};