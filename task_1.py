import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Customer reviews
corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]


# Create CountVectorizer with English stop words removed
vectorizer = CountVectorizer(stop_words='english')


# Convert text into Bag of Words matrix
X = vectorizer.fit_transform(corpus)


# Extract vocabulary
vocabulary = vectorizer.get_feature_names_out()

print("Vocabulary:")
print(vocabulary)


# Convert matrix into Pandas DataFrame
bow_matrix = pd.DataFrame(
    X.toarray(),
    columns=vocabulary
)

print("\nBag of Words Matrix:")
print(bow_matrix)