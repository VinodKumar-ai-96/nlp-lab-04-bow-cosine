import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Documents
documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]


# Search query
query = ["machine learning algorithms for data"]


# Create vectorizer
vectorizer = CountVectorizer(stop_words='english')


# Fit on documents
doc_vectors = vectorizer.fit_transform(documents)


# Transform query using the same vocabulary
query_vector = vectorizer.transform(query)


# Calculate cosine similarity
similarity_scores = cosine_similarity(
    query_vector,
    doc_vectors
)


# Extract scores
scores = similarity_scores[0]


# Rank documents from highest to lowest
ranking = sorted(
    enumerate(scores, start=1),
    key=lambda x: x[1],
    reverse=True
)


# Display results
print("Search Query:")
print(query[0])

print("\nDocument Ranking:")

for doc_number, score in ranking:
    print(f"Document {doc_number}: {score:.4f}")
    print(f"Text: {documents[doc_number - 1]}")
    print()