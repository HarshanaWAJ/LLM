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

- **Application UI:** [http://localhost:8000/](http://localhost:8000/) – the prompt input and video player are served directly from the root.
- **Raw animations (optional):** [http://localhost:8000/animations/](http://localhost:8000/animations/) shows the directory of available `.mp4` files.

> ⚠️ **Important:** do **not** navigate to `/docs` or `/redoc` – those are the API documentation pages and do not contain the animation matcher interface.

With the UI open, type intents like `Where is the bathroom?` or `Call the doctor` or `I am so happy`. Press **Generate**, and the corresponding `.mp4` animation from your library will auto-play on the screen.

> **Note:** the backend now includes a simple animation generator.  When training the model it will automatically create placeholder videos for any hints that are missing from `data/animations` (this is now the default output directory, so generated animations show up among the originals).  The generator no longer simply renders text; it draws a crude stick figure inside a thinking‑bubble with the prompt written above its head and animates the figure based on keywords such as "wave", "run", "jump", "dance" (also triggered by words like "love" or "happy"), and more.  You can extend the keyword map in `animation_generator.py` to suit your needs.  If you need generated animations to match the resolution or duration of an existing clip, pass its path as the ``template_path`` argument; the generator will try to copy those properties.  Both the command‑line and web interfaces will spawn a new `.mp4` file that matches your typed text, letting you build up a library of prompt‑specific clips over time.

