# RAG-GCP — PDF Question Answering API on Google Cloud Run

A containerized **Retrieval-Augmented Generation (RAG)** application that lets users upload PDF documents and ask questions grounded in their contents. Built with **FastAPI**, **LangChain**, **OpenAI**, and **FAISS**, with a GitHub Actions workflow for deploying the Docker image to **Google Cloud Run**.

> **Project scope:** A compact RAG API and cloud-deployment demonstration. The current implementation is not a production-hardened, persistent multi-instance document service.

## How it works

```text
PDF upload (POST /upload)
    → PyPDFLoader
    → RecursiveCharacterTextSplitter (1,000 characters; 150 overlap)
    → OpenAI embeddings
    → FAISS vector index (created or updated on local disk)

Question (POST /ask)
    → FAISS similarity retrieval (top 4 chunks)
    → Context-constrained prompt
    → OpenAI gpt-4o-mini (temperature 0)
    → Generated answer
```

The upload endpoint indexes the document and rebuilds the in-memory question-answering chain. Subsequent questions retrieve relevant chunks from the local FAISS index and pass them as context to the language model.

## Technology stack

| Layer | Implementation |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic |
| PDF extraction | LangChain `PyPDFLoader` / pypdf |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `OpenAIEmbeddings` |
| Vector search | FAISS (`faiss-cpu`) |
| RAG orchestration | LangChain Expression Language (LCEL) |
| Answer generation | OpenAI `gpt-4o-mini` |
| Packaging | Docker, Python 3.10 |
| CI/CD | GitHub Actions |
| Hosting | Google Cloud Run; Artifact Registry |

**Configuration:** `app/config.py` defines `MODEL_VERSION = "v1.0.0"`, `LLM_MODEL = "gpt-4o-mini"`, `EMBEDDING_MODEL = "text-embedding-3-small"`, `CHUNK_SIZE = 1000`, and `CHUNK_OVERLAP = 150`. Note that the current code instantiates `OpenAIEmbeddings()` without passing `EMBEDDING_MODEL` explicitly, so the configured embedding-model constant is **not currently wired into the embedding calls**.

## Repository structure

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py       # FastAPI routes and in-memory QA chain
│   ├── ingest.py     # PDF loading, chunking, embeddings, FAISS updates
│   ├── rag.py        # Retriever, prompt and LLM chain
│   ├── config.py     # Model and chunking configuration
│   └── storage.py    # Local upload and vector-index directories
├── .github/workflows/
│   └── deploy.yml    # Build, push and Cloud Run deployment
├── Dockerfile
├── requirements.txt
└── README.md
```

At runtime, `data/uploads/` stores uploaded PDFs and `vectorstore/` stores the local FAISS index.

## Run locally

**Prerequisites:** Python 3.10 or compatible, pip, and an OpenAI API key.

```bash
git clone https://github.com/Ali-00/RAG-GCP.git
cd RAG-GCP

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="your-openai-api-key"
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger API documentation.

### API endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health status and application model version |
| `POST` | `/upload` | Upload a PDF, index its text, and initialize the QA chain |
| `POST` | `/ask` | Ask a question using the currently initialized QA chain |

**1. Upload a PDF**

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@example.pdf"
```

Example response:

```json
{"status":"uploaded_and_indexed","file":"example.pdf"}
```

**2. Ask a question**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key findings in the document?"}'
```

The endpoint returns the question and the chain's result in an `answer` field. **Implementation note:** `build_qa_chain()` ends with `ChatOpenAI`, so `invoke()` produces an AI message object; the current route does not extract `.content`. For a clean string response, update the route to use `result_text.content` or append a string output parser to the chain.

Uploading a non-PDF filename returns HTTP 400. Asking before a document has been uploaded **in the current process** also returns HTTP 400.

## Docker

```bash
docker build -t rag-gcp .
docker run --rm -p 8080:8080 -e OPENAI_API_KEY="$OPENAI_API_KEY" rag-gcp
```

The Docker image runs Uvicorn on port **8080**. Access its API documentation at **http://localhost:8080/docs**.

## Google Cloud deployment

The existing `.github/workflows/deploy.yml` runs on pushes to `main` and:

1. Authenticates to Google Cloud using the `GCP_SA_KEY` GitHub secret.
2. Configures Docker authentication for Artifact Registry in `us-central1`.
3. Builds and pushes an image tagged with the Git commit SHA to `us-central1-docker.pkg.dev/<PROJECT_ID>/rag-app-repo/rag-app`.
4. Deploys the image as the **rag-app** Cloud Run service in **us-central1**.

The workflow references these GitHub Actions secrets: `GCP_SA_KEY`, `GCP_PROJECT`, `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, and `LANGCHAIN_TRACING_V2`. The Google Cloud project, Artifact Registry repository, service-account permissions, and relevant APIs must be configured before deployment.

**Security note:** The current workflow passes API credentials through `--set-env-vars`. For a hardened deployment, use Google Secret Manager with Cloud Run secret references and least-privilege service accounts. The workflow also specifies `--allow-unauthenticated`; protect the service before exposing sensitive documents.

## Current limitations and future improvements

- **Ephemeral storage:** PDFs and FAISS data are written to the container filesystem. Cloud Run instances do not provide durable shared storage, so data can disappear when instances restart or scale.
- **Process-local chain:** The QA chain starts as `None` and is initialized only after an upload in that process. Another instance may not be ready to answer even if a different instance indexed a PDF.
- **No document isolation:** New uploads add chunks to the existing index; there are no user, tenant, or document-level retrieval filters.
- **Upload validation:** The code checks the `.pdf` filename extension, but does not enforce file size limits, verify PDF content, or sanitize the uploaded filename.
- **FAISS deserialization:** `allow_dangerous_deserialization=True` is enabled. Only load indexes from trusted sources.
- **Response formatting:** Extract the AI message's text content for a predictable JSON answer.
- **Embedding configuration:** Pass the intended `EMBEDDING_MODEL` to `OpenAIEmbeddings(model=...)` consistently during ingestion and retrieval.
- **Observability and quality:** Add automated tests, retrieval/answer evaluations, structured logging, tracing, and error handling.

For a scalable next iteration, move document and index storage to durable managed services, load or initialize retrieval state independently of upload requests, and introduce authentication and per-document access controls.

## License

No license is declared in this README. Consult the repository for any separately added license file before reusing the code.
