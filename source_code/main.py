import os
import json
import pickle
import numpy as np
import random
from document import load_and_preprocess_documents, document_to_w2v, document_to_sbert
from recommendation import load_recommendation_data, build_knn_model, recommend_articles

# ✅ 控制是否启用分类训练
ENABLE_CLASSIFICATION = False

def prepare_recommendation_data(documents, labels, filenames):
    print("📌 Saving recommendation input files...")
    os.makedirs("data", exist_ok=True)

    # Word2Vec vectors
    vectors = np.array([document_to_w2v(doc) for doc in documents])
    np.save("data/article_vectors.npy", vectors)

    # 原始标题（备用）
    with open("data/article_titles.json", "w") as f:
        json.dump(filenames, f)

    # 使用原始标签
    with open("data/article_labels.json", "w") as f:
        json.dump(labels, f)

    # 保存原始文本
    with open("data/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("✅ Recommendation input files saved.")


def run_classification(documents, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import KFold
    import torch
    from classification import (
        train_naive_bayes_tfidf,
        train_knn_w2v,
        train_xgboost_tfidf,
        train_xgboost_w2v,
        train_neural_net_w2v,
        train_xgboost_sbert,
        train_neural_net_sbert
    )
    from util import compute_metrics, print_classification_results, plot_cross_validation, plot_confusion_matrix

    print("\n📌 Running Classification Evaluation...")
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y = np.array([label_to_index[label] for label in labels])

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(documents)
    X_w2v = np.array([document_to_w2v(doc) for doc in documents])
    X_sbert = document_to_sbert(documents)

    from run_classifier import run_all_classifiers
    results = run_all_classifiers(X_tfidf, X_w2v, X_sbert, y, unique_labels, index_to_label)

    model_names = {
        "nb": "Naive Bayes + TF-IDF",
        "knn": "kNN + Word2Vec",
        "xgb_tfidf": "XGBoost + TF-IDF",
        "xgb_w2v": "XGBoost + Word2Vec",
        "nn_w2v": "Neural Net + Word2Vec",
        "xgb_sbert": "XGBoost + SBERT",
        "nn_sbert": "Neural Net + SBERT"
    }

    for key, name in model_names.items():
        print(f"\n📌 Final Classification Report ({name}):")
        print_classification_results(results[key]["conf_mat"], results[key]["accuracies"], unique_labels)
        plot_cross_validation(results[key]["accuracies"], title=f"{name} Accuracy")
        plot_confusion_matrix(results[key]["conf_mat"], unique_labels, title=f"{name} Confusion Matrix")


def recommend_flow():
    vectors, titles, labels, documents = load_recommendation_data()
    knn_model = build_knn_model(vectors)

    # Step 1: 首页推荐
    candidate_indices = random.sample(range(len(titles)), 5)
    print("\n📚 以下是为你推荐的文章（随机挑选）：\n")
    for i, idx in enumerate(candidate_indices):
        print(f"{i + 1}. [{labels[idx]}] {titles[idx]}")

    choice = input("\n请输入你想阅读的文章编号 (1-5)，或按 Q 退出系统: ").strip()
    if choice.lower() == "q":
        print("👋 感谢使用，欢迎下次再来！")
        return
    selected_index = candidate_indices[int(choice) - 1]
    print(f"\n✅ 你选择阅读：[{labels[selected_index]}] {titles[selected_index]}\n")
    print("📖 正文内容如下：\n")
    print(documents[selected_index])

    # Step 2+: 无限推荐直到退出
    current_index = selected_index
    while True:
        recommended_indices = recommend_articles(knn_model, current_index)
        print("\n📢 你可能还喜欢这些文章：\n")
        for i, idx in enumerate(recommended_indices):
            print(f"{i + 1}. [{labels[idx]}] {titles[idx]}")

        next_choice = input("\n请输入你想继续阅读的文章编号 (1-5)，或按 Q 退出系统: ").strip()
        if next_choice.lower() == "q":
            print("👋 感谢阅读，再见！")
            break
        current_index = recommended_indices[int(next_choice) - 1]
        print(f"\n✅ 你选择阅读：[{labels[current_index]}] {titles[current_index]}\n")
        print("📖 正文内容如下：\n")
        print(documents[current_index])


def main():
    # 加载数据
    print("\n📌 Loading and Preprocessing Dataset...")
    data_dir = "../data/bbc/"
    documents, labels, filenames = load_and_preprocess_documents(data_dir)

    # 可选：训练分类模型
    if ENABLE_CLASSIFICATION:
        run_classification(documents, labels)

    # 保存推荐系统数据（如果文件不存在）
    if not os.path.exists("data/article_vectors.npy"):
        prepare_recommendation_data(documents, labels, filenames)

    # 进入推荐系统交互
    recommend_flow()


if __name__ == "__main__":
    main()
