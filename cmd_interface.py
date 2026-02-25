import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from animation_generator import create_animation, sanitize_hint

warnings.filterwarnings('ignore')

MODEL_DIR = "model"

def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading training embeddings and labels...")
    data = np.load(os.path.join(MODEL_DIR, "embeddings.npz"))
    train_embeddings = data['X']
    train_labels = data['labels']

    print("\n--- LLM Animation Matcher CMD Interface ---")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to close.")
    
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            # create an animation for exactly the user prompt as well; this
            # keeps a record of what was asked and lets people open the
            # generated video separately if desired.
            created = create_animation(user_input, "data/animations")
            if created:
                print(f"(generated new animation for prompt: {created})")

            # Embed
            emb = embedder.encode([user_input])
            
            # Find most similar training example
            similarities = cosine_similarity(emb, train_embeddings)[0]
            best_idx = np.argmax(similarities)
            pred_label = train_labels[best_idx]
            
            # Get video file
            animation_file = f"{pred_label}.mp4"
            print(f"-> Matched Animation: {animation_file}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
