from flask import Flask, request, jsonify, render_template
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Step 1: Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Step 2: Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 3: Apply LSA (using TruncatedSVD)
lsa = TruncatedSVD(n_components=100, random_state=42)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Step 4: Vectorize the query
    query_vector = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vector)

    # Step 5: Calculate cosine similarity between the query and documents
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]

    # Step 6: Sort documents by similarity and return the top 5
    top_indices = np.argsort(similarities)[::-1][:5]
    top_documents = [{"docId": f"Document {i+1}", 
                      "text": documents[i][:2000] + "...",  # Display a snippet of the document
                      "similarity": float(similarities[i])} for i in top_indices]

    return jsonify(top_documents)

if __name__ == '__main__':
    app.run(debug=True)
