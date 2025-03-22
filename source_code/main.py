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

    read_indices = set()  # ✅ 记录用户已读文章索引

    def show_random_articles():
        indices = random.sample(range(len(titles)), 5)
        print("\n📚 Here are 5 articles you might be interested in (random selection):\n")
        for i, idx in enumerate(indices):
            print(f"{i + 1}. [{labels[idx]}] {titles[idx]}")
        return indices

    # 👉 首页推荐循环
    while True:
        candidate_indices = show_random_articles()
        choice = input("\nEnter the article number you want to read (1-5), R to refresh, or Q to quit: ").strip()

        if choice.lower() == "q":
            print("👋 Thank you for using the system. See you next time!")
            return
        elif choice.lower() == "r":
            continue
        elif choice.isdigit() and 1 <= int(choice) <= 5:
            current_index = candidate_indices[int(choice) - 1]
            read_indices.add(current_index)
            print(f"\n✅ You chose to read: [{labels[current_index]}] {titles[current_index]}\n")
            print("📖 Article content:\n")
            print(documents[current_index])
        else:
            print("⚠️ Invalid input. Please try again.")
            continue  # 🚫 防止进入无限推荐循环

        # 👉 无限推荐循环
        while True:
            recommended_indices = recommend_articles(knn_model, current_index)

            print("\n📢 Based on your reading, we recommend the following articles:\n")
            shown_titles = set()
            displayed_indices = []
            for idx in recommended_indices:
                if idx in read_indices:
                    continue  # ✅ 不推荐已读
                title = titles[idx]
                if title not in shown_titles:
                    shown_titles.add(title)
                    displayed_indices.append(idx)
                    print(f"{len(displayed_indices)}. [{labels[idx]}] {title}")
                if len(displayed_indices) == 5:
                    break

            if not displayed_indices:
                print("⚠️ No new articles to recommend. Returning to homepage.")
                break

            # 🎯 推荐精准度
            target_label = labels[current_index]
            matched = sum(1 for idx in displayed_indices if labels[idx] == target_label)
            precision = matched / len(displayed_indices)
            print(f"\n🎯 Recommendation Precision: {matched} / {len(displayed_indices)} belong to the same category (Precision = {precision:.2f})")

            # 用户下一步选择
            user_input = input("\nEnter the article number you want to read (1-5), R to refresh, or Q to quit: ").strip()

            if user_input.lower() == "q":
                print("👋 Thank you for reading. Goodbye!")
                return
            elif user_input.lower() == "r":
                break
            elif user_input.isdigit() and 1 <= int(user_input) <= len(displayed_indices):
                current_index = displayed_indices[int(user_input) - 1]
                read_indices.add(current_index)
                print(f"\n✅ You chose to read: [{labels[current_index]}] {titles[current_index]}\n")
                print("📖 Article content:\n")
                print(documents[current_index])
            else:
                print("⚠️ Invalid input. Please try again.")

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
