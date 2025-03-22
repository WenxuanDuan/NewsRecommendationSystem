import numpy as np
import json
import pickle
from sklearn.neighbors import NearestNeighbors

def load_recommendation_data(vector_path="data/article_vectors.npy",
                             title_path="data/article_titles.json",
                             label_path="data/article_labels.json",
                             doc_path="data/documents.pkl"):
    vectors = np.load(vector_path)

    with open(title_path, "r") as f:
        filenames = json.load(f)

    with open(label_path, "r") as f:
        labels = json.load(f)

    with open(doc_path, "rb") as f:
        documents = pickle.load(f)

    titles = [doc.strip().replace("\n", " ")[:50] + "..." for doc in documents]

    return vectors, titles, labels, documents


def build_knn_model(vectors, metric="cosine", k=5):
    """
    构建 KNN 模型（用于推荐）
    """
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric)  # +1 是因为包含自己
    model.fit(vectors)
    return model


def recommend_articles(knn_model, article_index, top_k=5):
    distances, indices = knn_model.kneighbors([knn_model._fit_X[article_index]], n_neighbors=top_k + 1)
    unique_indices = []
    for idx in indices[0]:
        if idx != article_index and idx not in unique_indices:
            unique_indices.append(idx)
        if len(unique_indices) == top_k:
            break
    return unique_indices
