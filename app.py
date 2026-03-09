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
    model_path = os.path.join(MODEL_DIR, "finetuned_model")
    if os.path.exists(model_path):
        embedder = SentenceTransformer(model_path)
    else:
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

@app.post("/api/find", response_model=PredictResponse)
def find_animation(req: PredictRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    emb = embedder.encode([req.prompt])
    similarities = cosine_similarity(emb, train_embeddings)[0]
    best_idx = np.argmax(similarities)
    pred_label = train_labels[best_idx]
    
    # Return the mapped canonical animation filename
    animation_file = f"{pred_label}.mp4"
    return PredictResponse(animation_file=animation_file)

@app.post("/api/generate", response_model=PredictResponse)
def generate_new_animation(req: PredictRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Still find the best semantic match to use as the base motion
    emb = embedder.encode([req.prompt])
    similarities = cosine_similarity(emb, train_embeddings)[0]
    best_idx = np.argmax(similarities)
    pred_label = train_labels[best_idx]
    
    # generate a custom clip that explicitly matches the text supplied by the user
    # using the closest canonical video as the motion template
    from animation_generator import sanitize_hint
    create_animation(req.prompt, "data/generated")
    
    # return the newly generated custom video
    safe_name = sanitize_hint(req.prompt)
    animation_file = f"{safe_name}.mp4"
    return PredictResponse(animation_file=animation_file)

# Mount frontend files
# Videos served from simple paths; this keeps URIs short and avoids
# confusion about the data directory structure.
app.mount("/api/animations", StaticFiles(directory="data/animations"), name="animations")
os.makedirs("data/generated", exist_ok=True)
app.mount("/api/generated_animations", StaticFiles(directory="data/generated"), name="generated_animations")

# Serve the React/HTML interface directly at the root path.  Setting html=True
# makes the static mount return index.html for / automatically, so users only
# need to browse to http://<host>:<port>/ to see the UI.  This also prevents
# accidentally hitting FastAPI's documentation pages when looking for the
# prompt input.
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    # This allows running the app directly with `python app.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)

