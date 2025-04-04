import os
import json
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "source_code/data"

def load_article_data():
    """
    加载文章标题、标签、正文和向量
    """
    vectors = np.load(os.path.join(DATA_DIR, "article_vectors.npy"))

    with open(os.path.join(DATA_DIR, "original_article_titles.json")) as f:
        titles = json.load(f)

    with open(os.path.join(DATA_DIR, "article_labels.json")) as f:
        labels = json.load(f)

    with open(os.path.join(DATA_DIR, "original_documents.pkl"), "rb") as f:
        documents = pickle.load(f)

    return vectors, titles, labels, documents


def get_summary(text, max_lines=2):
    """
    获取文章前几行（摘要）
    """
    lines = text.strip().split("\n")
    flat = " ".join(lines)
    return " ".join(flat.split()[:40]) + "..."  # 限前40个词


def build_knn_model(vectors, n_neighbors=10):
    """
    构建 KNN 模型用于推荐
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn.fit(vectors)
    return knn


def recommend_articles(knn_model, current_index, read_indices=None, top_k=5):
    """
    根据当前文章索引推荐 top_k 个相似文章（排除已读）
    """
    distances, indices = knn_model.kneighbors([knn_model._fit_X[current_index]])
    indices = indices[0]

    result = []
    for idx in indices:
        if idx == current_index:
            continue
        if read_indices and idx in read_indices:
            continue
        result.append(idx)
        if len(result) == top_k:
            break
    return result
