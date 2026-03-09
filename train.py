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

    print("Finetuning model...")
    try:
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
        
        train_examples = []
        for text, label in zip(valid_texts, valid_target_labels):
            # We want the prompt to be close to its canonical animation label
            label_text = label.replace("_", " ")
            train_examples.append(InputExample(texts=[text, label_text]))
            
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(model=embedder)
        
        print("Starting training...")
        # Adjust epochs as needed
        embedder.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=10)
    except Exception as e:
        print(f"Warning: Finetuning failed or is not supported (error: {e}). Proceeding without finetuning.")

    print("Generating canonical label embeddings...")
    # Get unique labels to act as targets
    unique_labels = sorted(list(set(valid_target_labels)))
    unique_labels_texts = [l.replace("_", " ") for l in unique_labels]
    label_embeddings = embedder.encode(unique_labels_texts, show_progress_bar=True)

    print("Saving embeddings...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    embedder.save(os.path.join(MODEL_DIR, "finetuned_model"))
    np.savez(os.path.join(MODEL_DIR, "embeddings.npz"), X=label_embeddings, labels=np.array(unique_labels))

    print("Evaluating nearest neighbor...")
    correct = 0
    # Evaluate accuracy on the tuning dataset to see if it mapped them to the right canonical label
    eval_embeddings = embedder.encode(valid_texts, show_progress_bar=False)
    for i in range(len(valid_texts)):
        similarities = cosine_similarity([eval_embeddings[i]], label_embeddings)[0]
        best_idx = np.argmax(similarities)
        pred_label = unique_labels[best_idx]
        if pred_label == valid_target_labels[i]:
            correct += 1

    accuracy = correct / len(valid_texts)
    print(f"Accuracy mapping text to animation labels: {accuracy * 100:.2f}%")
    print("Done!")


if __name__ == "__main__":
    main()
