# Running the AI Animation Matcher Project

Welcome to the AI Animation Matcher! This guide explains how to get the application up and running.

## 1. Setup Environment
First, you need to verify your python environment and dependencies. The virtual environment setup (`venv`) is located in the `e:\Projects\LLM` directory.
In your command line or PowerShell, run:
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the Model
The AI model utilizes `sentence-transformers` for robust embedding generation and a `scikit-learn` classifier for gesture matching. These models process your dataset and cache their states into the `model/` folder.
You only need to run this command **once** before starting the inference engines or whenever your dataset (`data/dataset.jsonl`) is modified:
```powershell
python train.py
```

## 3. Options for Running the AI
You have two options to interact with the model: from a fast terminal or a modern web UI.

### Option A: Command-Line Interface (CMD)
A fast, lightweight interface allowing you to chat with the model inside your terminal. It returns the name of the corresponding video.
Run:
```powershell
python cmd_interface.py
```
Type prompts into the interface and hit `Enter`. Type `exit` to close it.

### Option B: The Application / Web UI
We have a backend REST API wrapper running in FastAPI (`app.py`), connected to our modern Web UI (`static/index.html`).

First, launch the server processing engine:
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```
*(If you are already running this command somewhere else, you don't need to run it again!)*

Once it prints `Application startup complete.`, open any Web Browser (e.g. Chrome, Edge) and go to:
[http://localhost:8000](http://localhost:8000)

Here you will see a sleek web application. There's an input box near the bottom. Type intents like `Where is the bathroom?` or `Call the doctor` or `I am so happy` into it. Press `Generate`, and the precise `.mp4` animation from your local library will auto-play on the screen.

