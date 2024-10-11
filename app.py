from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample documents (you should replace these with your actual dataset)
documents = [
    "NASA plans to build a new space station.",
    "SpaceX launches a new rocket to Mars.",
    "The International Space Station has been in orbit for 20 years.",
    "The moon landing was one of NASA's greatest achievements.",
    "Mars exploration is a priority for space agencies.",
    "Space tourism is becoming increasingly popular.",
    "The Hubble Space Telescope captures stunning images of space."
]

# Step 1: Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 2: Apply LSA (using TruncatedSVD)
lsa = TruncatedSVD(n_components=2)  # Reduce to 2 dimensions for simplicity
lsa_matrix = lsa.fit_transform(tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Step 3: Vectorize the query
    query_vector = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vector)

    # Step 4: Calculate cosine similarity between the query and documents
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]

    # Step 5: Sort documents by similarity
    top_indices = np.argsort(similarities)[::-1][:5]
    top_documents = [{"docId": f"Document {i+1}", "text": documents[i], "similarity": float(similarities[i])} for i in top_indices]

    return jsonify(top_documents)

if __name__ == '__main__':
    app.run(debug=True)
