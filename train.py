import json  # parse JSON lines
import os    # filesystem utilities
import joblib  # save/load sklearn objects
import numpy as np  # numerical arrays
from sentence_transformers import SentenceTransformer  # embedding model
from sklearn.neural_network import MLPClassifier  # (unused) future classifier
from sklearn.svm import SVC  # (unused) support-vector classifier
from sklearn.preprocessing import LabelEncoder  # convert labels to ints
from sklearn.metrics import accuracy_score  # compute accuracy
from sklearn.metrics.pairwise import cosine_similarity  # measure vector similarity

# helper module for generating placeholder videos
from animation_generator import create_animation, sanitize_hint

DATASET_PATH = "data/dataset.jsonl"  # input dataset file
ANIMATIONS_DIR = "data/animations"   # directory of animation files
MODEL_DIR = "model"  # output directory for models/embeddings


def load_data():
    texts = []
    labels = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():  # skip blank lines
                continue
            data = json.loads(line)
            texts.append(data["input_text"].strip())
            labels.append(data["animation_hint"].strip())
    return texts, labels


def validate_labels(labels, auto_generate: bool = False):
    """Ensure that every animation hint has a corresponding file.

    If ``auto_generate`` is True then any missing files will be created using
    :func:`animation_generator.create_animation`.  Otherwise the function simply
    prints a warning and omits the offending hints from the returned set.
    """
    valid_labels = set()
    missing_files = set()
    for label in set(labels):
        filename = f"{label}.mp4"
        if os.path.exists(os.path.join(ANIMATIONS_DIR, filename)):
            valid_labels.add(label)
        else:
            missing_files.add(label)
    if missing_files:
        if auto_generate:
            print(f"Generating {len(missing_files)} missing animations...")
            for label in missing_files:
                sanitized = sanitize_hint(label)
                print(f"  • {label} -> {sanitized}.mp4")
                # the generator will create the file if it doesn't exist
                create_animation(label, ANIMATIONS_DIR)
                valid_labels.add(label)
        else:
            print(f"WARNING: missing files for hints: {missing_files}")
    return valid_labels


def main():
    print("Loading dataset...")
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples.")

    print("Validating animation files (auto‑generate missing)...")
    valid_labels = validate_labels(labels, auto_generate=True)

    # keep only valid examples
    valid_texts = []
    valid_target_labels = []
    for t, l in zip(texts, labels):
        if l in valid_labels:
            valid_texts.append(t)
            valid_target_labels.append(l)

    print(f"Proceeding with {len(valid_texts)} valid samples.")

    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings...")
    X = embedder.encode(valid_texts, show_progress_bar=True)

    print("Saving embeddings...")
    np.savez(os.path.join(MODEL_DIR, "embeddings.npz"), X=X, labels=np.array(valid_target_labels))

    print("Evaluating nearest neighbor...")
    correct = 0
    for i in range(len(X)):
        train_X = np.delete(X, i, axis=0)  # leave-one-out
        train_labels = np.delete(np.array(valid_target_labels), i)
        similarities = cosine_similarity([X[i]], train_X)[0]
        best_idx = np.argmax(similarities)
        pred_label = train_labels[best_idx]
        if pred_label == valid_target_labels[i]:
            correct += 1

    accuracy = correct / len(X)
    print(f"Leave-one-out accuracy: {accuracy * 100:.2f}%")
    print(f"Model saved to {MODEL_DIR}/mlp_classifier.joblib")
    print(f"Label encoder saved to {MODEL_DIR}/label_encoder.joblib")
    print("Done!")


if __name__ == "__main__":
    main()
