from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from animation_generator import create_animation
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Animation Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "model"
# Load models globally
print("Loading models for API...")
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    data = np.load(os.path.join(MODEL_DIR, "embeddings.npz"))
    train_embeddings = data['X']
    train_labels = data['labels']
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please make sure you have run train.py first.")

class PredictRequest(BaseModel):
    prompt: str

class PredictResponse(BaseModel):
    animation_file: str

@app.post("/api/predict", response_model=PredictResponse)
def predict_animation(req: PredictRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    emb = embedder.encode([req.prompt])
    similarities = cosine_similarity(emb, train_embeddings)[0]
    best_idx = np.argmax(similarities)
    pred_label = train_labels[best_idx]
    
    # generate a clip that exactly matches the text supplied by the user
    # so that a record exists in the animation directory
    create_animation(req.prompt, "data/animations")

    # Return the mapped animation filename
    animation_file = f"{pred_label}.mp4"
    return PredictResponse(animation_file=animation_file)

# Mount frontend files
# Videos served from a simple /animations path; this keeps URIs short and avoids
# confusion about the data directory structure.
app.mount("/animations", StaticFiles(directory="data/animations"), name="animations")

# Serve the React/HTML interface directly at the root path.  Setting html=True
# makes the static mount return index.html for / automatically, so users only
# need to browse to http://<host>:<port>/ to see the UI.  This also prevents
# accidentally hitting FastAPI's documentation pages when looking for the
# prompt input.
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

