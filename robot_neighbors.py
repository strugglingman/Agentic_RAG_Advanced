import os, sys, glob, pickle, argparse, textwrap
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, ~22MB
INDEX_FILE = "index.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


def read_texts(file_path):
    filename = os.path.splitext(file_path)
    ext = filename[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = start + chunk_size - chunk_overlap

    return chunks

def ingest(file_path):
    text = read_texts(file_path)
    chunks = chunk_text(text)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    nn = NearestNeighbors(n_neighbors=min(TOP_K, len(chunks)), metric="cosine")
    nn.fit(embeddings)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "embeddings": embeddings,
            "nn": nn,
            "model_name": EMBED_MODEL_NAME
        }, f
        )

def load_index():
    if not os.path.exists(INDEX_FILE):
        print("Index file not found. Please run the script with --ingest <file_path> to create the index.")
        sys.exit(1)
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)

def retrieve(query, top_k=TOP_K):
    data = load_index()
    model = SentenceTransformer(data["model_name"])
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    nn: NearestNeighbors = data["nn"]
    distance, indices = nn.kneighbors(qvec, n_neighbors=min(top_k, len(data["chunks"])))
    idxs = indices[0]
    hits = []
    for i in idxs:
        hits.append({
            "chunk": data["chunks"][i],
        })

    return hits

def build_prompt(query, ctx):
    context_str = "\n\n".join(f"Context {i+1}:\n{hit['chunk']}" for i, hit in enumerate(ctx))
    system = ("You are a helpful assistant that answers questions based on the provided context. "
              "If the answer is not contained within the context, respond with 'I don't know.'\n")
    
    user = f"Question: {query}\n\n, and answer the question based on the context: {context_str}\n\nAnswer:"

    return system, user

def chat():
    #load_dotenv()
    #api_key = os.getenv("OPENAI_API_KEY")
    #if not api_key:
    #    print("Please set the OPENAI_API_KEY environment variable.")
    #    sys.exit(1)

  # Load API key from environment; fail fast if missing.
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    print("OPENAI_API_KEY not set; export it before running chat().")
    return
  client = OpenAI(api_key=api_key)
    while True:
        q = input("> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        ctx = retrieve(q, top_k=TOP_K)
        system, user = build_prompt(q, ctx)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=512,
            temperature=0.2,
        )

        print(system)
        print(user)
        answer = response.choices[0].message.content
        print("\n" + answer + "\n")

#data = load_index()
#chat()
#ingest("d:\\learning\\robot\\skolan.pdf")

# ----------------- Routes -----------------

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Mini RAG – Chat with Your Files</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 2rem; max-width: 900px; }
    header { margin-bottom: 1rem; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
    .row { display: flex; gap: 1rem; flex-wrap: wrap; }
    label { font-weight: 600; display:block; margin-bottom: .5rem; }
    input[type="text"], textarea { width: 100%; padding: .6rem; border: 1px solid #ccc; border-radius: 8px; }
    button { padding: .6rem 1rem; border: 0; border-radius: 8px; background: #0d6efd; color: #fff; cursor: pointer; }
    button.secondary { background: #6c757d; }
    .muted { color: #666; font-size: .9rem; }
    .sources { color:#444; font-size: .95rem; }
    .status { padding:.5rem .75rem; border-radius:8px; background:#f8f9fa; border:1px solid #eee; }
    .success { border-color:#c7f3d0; background:#e9fbea; }
    .error { border-color:#f5c2c7; background:#fcebea; }
    code { background:#f6f8fa; padding:.15rem .3rem; border-radius:6px; }
  </style>
</head>
<body>
<header>
  <h1>Mini RAG – Chat with Your Files</h1>
  <p class="muted">Upload .txt / .md / .pdf, build an index, then ask questions grounded in those files.</p>
</header>

<div class="card">
  <h2>1) Upload files</h2>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <label>Select files (.txt, .md, .pdf)</label>
    <input type="file" name="files" multiple />
    <p class="muted">Files are saved to <code>./uploads</code>.</p>
    <button type="submit">Upload</button>
  </form>
</div>

<div class="card">
  <h2>2) Build / Rebuild index</h2>
  <form method="post" action="/ingest">
    <button type="submit">Build index from uploads</button>
    <span class="muted">Embeddings: {{ model_name }}. Index file: <code>{{ index_file }}</code></span>
  </form>
</div>

<div class="card">
  <h2>3) Ask a question</h2>
  <form method="post" action="/ask">
    <label>Your question</label>
    <input name="q" type="text" placeholder="e.g., What is our refund policy?" />
    <button type="submit">Ask</button>
  </form>
</div>

{% if msg %}
  <div class="card status {{ 'success' if ok else 'error' }}">
    <strong>{{ 'OK' if ok else 'Error' }}:</strong> {{ msg }}
  </div>
{% endif %}

{% if answer %}
  <div class="card">
    <h3>Answer</h3>
    <div>{{ answer|safe }}</div>
  </div>
  {% if sources %}
  <div class="card">
    <h3>Sources</h3>
    <div class="sources">
      <ul>
        {% for s in sources %}
          <li>{{ s }}</li>
        {% endfor %}
      </ul>
    </div>
  </div>
  {% endif %}
{% endif %}

<footer class="muted">
  <p>Built with Flask + SentenceTransformers + scikit-learn + OpenAI. Overlap keeps context across chunk boundaries.</p>
</footer>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    data = load_index()
    model_name = data["model_name"] if data else EMBED_MODEL_NAME
    return render_template_string(HTML, msg=None, ok=True, answer=None, sources=None,
                                  model_name=model_name, index_file=INDEX_FILE)

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    saved = []
    for f in files:
        if not f or f.filename == "":
            continue
        name = secure_filename(f.filename)
        if not allowed(name):
            continue
        path = os.path.join(UPLOAD_DIR, name)
        f.save(path)
        saved.append(name)
    msg = f"Uploaded {len(saved)} file(s): {', '.join(saved) or 'None'}"
    return render_template_string(HTML, msg=msg, ok=True, answer=None, sources=None,
                                  model_name=EMBED_MODEL_NAME, index_file=INDEX_FILE)

@app.route("/ingest", methods=["POST"])
def ingest():
    info, err = build_index_from_folder(UPLOAD_DIR)
    if err:
        return render_template_string(HTML, msg=err, ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file=INDEX_FILE)
    msg = f"Indexed {info['chunks']} chunks from {info['files']} files."
    return render_template_string(HTML, msg=msg, ok=True, answer=None, sources=None,
                                  model_name=EMBED_MODEL_NAME, index_file=INDEX_FILE)

@app.route("/ask", methods=["POST"])
def ask():
    q = (request.form.get("q") or "").strip()
    if not q:
        return render_template_string(HTML, msg="Please enter a question.", ok=False,
                                      answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file=INDEX_FILE)
    ctx, err = retrieve(q, TOP_K)
    if err:
        return render_template_string(HTML, msg=err, ok=False, answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file=INDEX_FILE)

    system, user = build_prompt(q, ctx)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.2,
            max_tokens=500
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        return render_template_string(HTML, msg=f"OpenAI error: {e}", ok=False,
                                      answer=None, sources=None,
                                      model_name=EMBED_MODEL_NAME, index_file=INDEX_FILE)

    sources = sorted({os.path.basename(c["source"]) for c in ctx})
    return render_template_string(HTML, msg=None, ok=True, answer=answer, sources=sources,
                                  model_name=EMBED_MODEL_NAME, index_file=INDEX_FILE)

if __name__ == "__main__":
    app.run(debug=True)