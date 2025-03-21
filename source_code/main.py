import numpy as np
import torch
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from document import load_and_preprocess_documents, document_to_w2v, document_to_sbert
from util import compute_metrics, plot_confusion_matrix, plot_cross_validation, print_classification_results
from run_classifier import run_all_classifiers

def main():
    print("\n📌 Loading and Preprocessing Dataset...")
    data_dir = "../data/bbc/"
    documents, labels, filenames = load_and_preprocess_documents(data_dir)

    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y = np.array([label_to_index[label] for label in labels])

    print("\n📌 Training TF-IDF model...")
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(documents)

    print("\n📌 Computing Word2Vec features...")
    X_w2v = np.array([document_to_w2v(doc) for doc in documents])

    print("\n📌 Computing SBERT features...")
    X_sbert = document_to_sbert(documents)

    print("\n📌 Running 10-Fold Cross Validation...")
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

    # 📌 保存推荐系统输入数据
    print("\n📌 Saving Recommendation System Input Files...")
    np.save("../data/article_vectors.npy", X_w2v)
    with open("../data/article_titles.json", "w") as f:
        json.dump(filenames, f)

    xgb_model = results["xgb_w2v"].get("model")
    if xgb_model is not None:
        predicted_labels = xgb_model.predict(X_w2v).tolist()
        with open("../data/article_labels.json", "w") as f:
            json.dump(predicted_labels, f)
        print("✅ article_labels.json saved.")
    else:
        print("⚠️ Warning: xgb_w2v model not found in results.")

if __name__ == "__main__":
    main()