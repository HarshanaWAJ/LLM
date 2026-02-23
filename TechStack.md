1️⃣ Sentence Embeddings (Transformer-based NLP)

Library: sentence-transformers
Model: all-MiniLM-L6-v2

What it is:

You are using a pretrained transformer model to convert text into numerical vectors (embeddings).

SentenceTransformer('all-MiniLM-L6-v2') loads a compact transformer model trained to generate semantically meaningful sentence embeddings.

Why it's important:

Instead of using traditional features (like TF-IDF or bag-of-words), this model:

Captures semantic meaning

Understands context

Places similar sentences close together in vector space

For example:

"Wave hello"

"Say hi with your hand"

These would produce similar embeddings.

Technology behind it:

Based on transformer architecture (similar to BERT)

Pretrained using contrastive learning

Produces 384-dimensional dense vectors

2️⃣ Vector Similarity (Cosine Similarity)

Library: sklearn.metrics.pairwise.cosine_similarity

What it does:

Measures how similar two embedding vectors are.

Formula intuition:

If vectors point in the same direction → similarity ≈ 1

If unrelated → similarity ≈ 0

If opposite → similarity ≈ -1

You use this for nearest neighbor matching:

similarities = cosine_similarity([X[i]], train_X)[0]
best_idx = np.argmax(similarities)

So the prediction is:

“Find the most semantically similar sentence in the dataset and use its animation label.”

This is a retrieval-based approach, not a trained classifier.

3️⃣ Nearest Neighbor Classification (Instance-Based Learning)

Instead of training a classifier (even though MLPClassifier and SVC are imported), you're using:

🔹 k-Nearest Neighbor logic (k=1)

For each sample:

Remove it from training set

Find the most similar embedding

Assign that label

This is:

Non-parametric

Lazy learning

Memory-based classification

4️⃣ Leave-One-Out Cross Validation (LOOCV)

You evaluate performance by:

For each sample:

Remove it

Predict it using the rest

Count correct predictions

Accuracy:

accuracy = correct / len(X)

LOOCV is:

Computationally expensive

Very reliable for small datasets

Maximizes training data usage

5️⃣ NumPy (Scientific Computing)

Used for:

Storing embeddings (np.array)

Removing elements (np.delete)

Saving embeddings (np.savez)

np.savez() creates a compressed .npz file containing:

Embedding matrix

Corresponding labels

This allows fast loading for future inference.

6️⃣ Scikit-learn (Machine Learning Utilities)

You import:

MLPClassifier

SVC

LabelEncoder

accuracy_score

But in this version of the script:

They are not actually used

The system works purely via similarity matching

You could later replace nearest neighbor with:

MLP neural network classifier

Support Vector Machine

Or hybrid approaches

7️⃣ Joblib (Model Serialization)

Imported but not used here.

Typically used for:

Saving trained models

Saving label encoders

Fast loading for deployment

Example:

joblib.dump(model, "model.joblib")
8️⃣ Data Validation Logic

Your script also includes:

File-based label validation
os.path.exists(os.path.join(ANIMATIONS_DIR, filename))

This ensures:

Only labels with corresponding .mp4 files are used

Prevents runtime failures

This is a practical data integrity safeguard.

🔎 What Type of System Is This?

Your pipeline is a:

Semantic retrieval-based animation selector

More specifically:

Text → Embedding

Compare against dataset

Return most similar animation label

It is not training a classifier, even though it prints model-saving messages.