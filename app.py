from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
    
    # Return the mapped animation filename
    animation_file = f"{pred_label}.mp4"
    return PredictResponse(animation_file=animation_file)

# Mount frontend files
app.mount("/data/animations", StaticFiles(directory="data/animations"), name="animations")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html not found in static folder. Please build the UI."}
