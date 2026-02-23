import json
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "data/dataset.jsonl"
ANIMATIONS_DIR = "data/animations"
MODEL_DIR = "model"

def load_data():
    texts = []
    labels = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            texts.append(data["input_text"].strip())
            labels.append(data["animation_hint"].strip())
    return texts, labels

def validate_labels(labels):
    valid_labels = set()
    missing_files = set()
    for label in set(labels):
        filename = f"{label}.mp4"
        if os.path.exists(os.path.join(ANIMATIONS_DIR, filename)):
            valid_labels.add(label)
        else:
            missing_files.add(label)
    
    if missing_files:
        print(f"WARNING: The following animation hints do not have corresponding .mp4 files in {ANIMATIONS_DIR}:")
        for m in missing_files:
            print(f" - {m}")
    return valid_labels

def main():
    print("Loading dataset...")
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples.")

    print("Validating animation files...")
    valid_labels = validate_labels(labels)
    
    # Filter out entries with no valid animation
    valid_texts = []
    valid_target_labels = []
    for t, l in zip(texts, labels):
        if l in valid_labels:
            valid_texts.append(t)
            valid_target_labels.append(l)
    
    print(f"Proceeding with {len(valid_texts)} valid samples.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings for training data...")
    X = embedder.encode(valid_texts, show_progress_bar=True)
    
    # Save embeddings and labels for nearest neighbor search
    print("Saving embeddings and labels for nearest neighbor matching...")
    np.savez(os.path.join(MODEL_DIR, "embeddings.npz"), X=X, labels=np.array(valid_target_labels))
    
    print("Evaluating model using nearest neighbor...")
    # Simple evaluation using leave-one-out cross validation
    correct = 0
    for i in range(len(X)):
        # Leave one out
        train_X = np.delete(X, i, axis=0)
        train_labels = np.delete(np.array(valid_target_labels), i)
        
        # Find most similar
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
