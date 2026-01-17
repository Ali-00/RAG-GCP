from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.rag import build_qa_chain
from app.storage import UPLOAD_DIR
from app.config import MODEL_VERSION
import shutil

app = FastAPI(
    title="Production RAG API",
    version=MODEL_VERSION
)

qa_chain = None

class AskRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_pdf(file_path.as_posix())

    global qa_chain
    qa_chain = build_qa_chain()

    return {
        "status": "uploaded_and_indexed",
        "file": file.filename
    }

@app.post("/ask")
def ask(req: AskRequest):
    global qa_chain
    if not qa_chain:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF first."
        )

    result_text = qa_chain.invoke(req.question)

    return {
        "question": req.question,
        "answer": result_text
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }
