# How to Run

This guide explains how to set up the environment, finetune the embedding model, and run the animation matcher API.

## 1. Prerequisites

First, ensure you have installed all the necessary dependencies. The `requirements.txt` file contains all the required libraries including `sentence-transformers`, `mediapipe`, `moviepy`, and `fastapi`.

```bash
pip install -r requirements.txt
```

> **Note**: Due to compatibility issues with some packages (like `moviepy`), you need to ensure `numpy` version is `<2.0.0`. If you face issues, run: `pip install "numpy<2"`.

## 2. Training and Finetuning the Model

Before running the API, you must generate the embeddings and finetune the model on your dataset.

```bash
python train.py
```

**What `train.py` does:**
1. Loads the dataset from `data/dataset.jsonl`.
2. Validates that all canonical animations in `data/animations/` exist. If any are missing, it uses `animation_generator.py` (and MediaPipe) to auto-generate realistic stick-figure videos for them.
3. Finetunes the `all-MiniLM-L6-v2` SentenceTransformer model using `MultipleNegativesRankingLoss` so that text prompts are mapped closer to their canonical animation labels.
4. Saves the finetuned model to the `model/finetuned_model/` directory.
5. Generates the final baseline embeddings and saves them to `model/embeddings.npz`.
6. Evaluates the model accuracy.

## 3. Running the Application

Once the model is tuned and embeddings are saved, you can start the API server.

```bash
python app.py
```
*(Alternatively: `uvicorn app:app --host 127.0.0.1 --port 8000`)*

**What `app.py` does:**
1. Loads the finetuned model from `model/finetuned_model/` and the embeddings from `model/embeddings.npz`.
2. Starts a FastAPI server on `http://localhost:8000`.
3. Serves the React/HTML frontend interface at the root URL `/`.
4. Exposes the `/api/predict` endpoint to process text prompts, find the closest semantic animation match, and generate new dynamically generated video artifacts if needed.
5. Serves generated animation videos from the `data/animations/` directory at the `/animations` path.

## 4. Testing the Animation Generator

If you want to test the realistic 2D skeleton generation manually without the API, you can run:

```bash
python test_anim.py
```

This will run a sample prompt through `animation_generator.py`'s MediaPipe Pose pipeline and output the generated video to `data/generated/`.
