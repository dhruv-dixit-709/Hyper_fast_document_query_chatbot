import os
import time
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from llama_cpp import Llama
import PyPDF2
import docx
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

knowledge_dir = Path("knowledge_files")
knowledge_dir.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


llm = Llama(model_path = "./hyper_tuned_model.gguf", 
            n_gpu_layers = 35,
            ctx_size = 2048)

processed_files = set()

def extract_text(file_path):
    if file_path.suffix == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    elif file_path.suffix == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.suffix == ".txt":
        return file_path.read_text()
    return ""

def process_new_files():
    while True:
        for file in knowledge_dir.iterdir():
            if file.name not in processed_files and file.suffix in {".pdf", ".docx", ".txt"}:
                text = extract_text(file)
                chunks = text_splitter.split_text(text)
                embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
                for i, chunk in enumerate(chunks):
                    collection.add(documents=[chunk], ids=[f"{file.name}_{i}"], embeddings=[embeddings[i]])
                processed_files.add(file.name)
        time.sleep(5)

@app.post("/upload")
async def upload_file(file: UploadFile):
    file_path = knowledge_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"message": f"{file.filename} uploaded and queued for processing."}

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_file(data: QueryRequest):
    query_embedding = embedding_model.embed_query(data.question)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_text = "\n".join(results["documents"][0])
    prompt = f"""
    You are an AI assistant that answers questions based on a provided document.
    If the answer is not in the document, say \"I don't know.\"

    Document:
    {retrieved_text}

    Question: {data.question}

    Based on the document, provide a detailed answer:
    """
    response = llm(prompt, max_tokens=512, temperature=0.1, top_k=10, top_p=0.5)
    return {"answer": response["choices"][0]["text"].strip()}

@app.get("/")
async def main_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Knowledge Chatbot</title></head>
    <body>
        <h2>Upload Document or Paste Text</h2>
        <form id="uploadForm">
            <input type="file" id="fileInput" name="file"><br><br>
            <button type="submit">Add to Knowledge Base</button>
        </form>

        <h2>Ask a Question</h2>
        <textarea id="question" rows="4" cols="60"></textarea><br>
        <button onclick="askQuestion()">Submit</button>
        <p><strong>Answer:</strong></p>
        <p id="answer"></p>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                await fetch('/upload', { method: 'POST', body: formData });
                alert("File uploaded successfully.");
            });

            async function askQuestion() {
                const question = document.getElementById("question").value;
                const response = await fetch("/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                document.getElementById("answer").innerText = data.answer;
            }
        </script>
    </body>
    </html>
    """)

# Start background file watcher
threading.Thread(target=process_new_files, daemon=True).start()
