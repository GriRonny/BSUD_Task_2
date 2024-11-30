import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np

# Load preprocessed reviews
df_processed = pd.read_csv('Data/reviews_preprocessed.csv')

# Load VADER sentiment results
df_vader = pd.read_csv('Data/reviews_VADER.csv')

# Combine DataFrames
df_combined = pd.concat([df_processed, df_vader[['vader_score', 'vader_label']]], axis=1)
df_combined.reset_index(drop=True, inplace=True)

# Separate positive and negative reviews
positive_reviews = df_combined[df_combined['vader_label'] == 'positive']['processed_reviews']
negative_reviews = df_combined[df_combined['vader_label'] == 'negative']['processed_reviews']

# Positive reviews vectorization
vectorizer_positive = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
X_positive = vectorizer_positive.fit_transform(positive_reviews)

# Negative reviews vectorization
vectorizer_negative = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
X_negative = vectorizer_negative.fit_transform(negative_reviews)

# Dimensionality Reduction
n_components = 100

# Positive reviews
svd_positive = TruncatedSVD(n_components=n_components, random_state=42)
X_positive_reduced = svd_positive.fit_transform(X_positive)

# Negative reviews
svd_negative = TruncatedSVD(n_components=n_components, random_state=42)
X_negative_reduced = svd_negative.fit_transform(X_negative)

# Clustering
k = 8

# Positive reviews
kmeans_positive = KMeans(n_clusters=k, random_state=42)
labels_positive = kmeans_positive.fit_predict(X_positive_reduced)

# Negative reviews
kmeans_negative = KMeans(n_clusters=k, random_state=42)
labels_negative = kmeans_negative.fit_predict(X_negative_reduced)

# Assign cluster labels
df_positive = df_combined[df_combined['vader_label'] == 'positive'].copy()
df_positive['cluster'] = labels_positive

df_negative = df_combined[df_combined['vader_label'] == 'negative'].copy()
df_negative['cluster'] = labels_negative


# Function to get top terms per cluster
def get_top_terms_per_cluster(tfidf_matrix, labels, vectorizer, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}

    for cluster_num in np.unique(labels):
        cluster_indices = np.where(labels == cluster_num)
        mean_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
        top_indices = np.argsort(mean_tfidf.A1)[::-1][:n_terms]
        top_terms = [terms[i] for i in top_indices]
        cluster_terms[cluster_num] = top_terms

    return cluster_terms


# Top terms in positive clusters
top_terms_positive = get_top_terms_per_cluster(X_positive, labels_positive, vectorizer_positive)
print("Top Terms in Positive Review Clusters:")
for cluster_num, terms in top_terms_positive.items():
    print(f"\nCluster {cluster_num}: {', '.join(terms)}")

# Top terms in negative clusters
top_terms_negative = get_top_terms_per_cluster(X_negative, labels_negative, vectorizer_negative)
print("\nTop Terms in Negative Review Clusters:")
for cluster_num, terms in top_terms_negative.items():
    print(f"\nCluster {cluster_num}: {', '.join(terms)}")
