import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from document import load_and_preprocess_documents, document_to_w2v
from util import compute_metrics, plot_confusion_matrix, plot_cross_validation, print_classification_results

def main():
    print("📌 Loading and Preprocessing Dataset...\n")
    data_dir = "../data/bbc/"
    documents, labels = load_and_preprocess_documents(data_dir)

    # **📌 处理 labels**
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y = np.array([label_to_index[label] for label in labels])

    # **📌 计算 TF-IDF 特征**
    print("\n📌 Training TF-IDF model...\n")
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(documents)

    # **📌 计算 Word2Vec 特征**
    print("\n📌 Computing Word2Vec features...\n")
    X_w2v = np.array([document_to_w2v(doc) for doc in documents])

    # **📌 10-Fold 交叉验证**
    print("\n📌 Running 10-Fold Cross Validation...\n")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 🔹 Naive Bayes
    nb_accuracies, nb_conf_mat = [], np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    # 🔹 kNN
    knn_accuracies, knn_conf_mat = [], np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tfidf), 1):
        print(f"🔹 Fold {fold}/10")

        # **📌 划分训练集 & 测试集**
        X_train_tfidf, X_test_tfidf = X_tfidf[train_idx], X_tfidf[test_idx]
        X_train_w2v, X_test_w2v = X_w2v[train_idx], X_w2v[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # **📌 训练 Naive Bayes**
        nb_model = MultinomialNB()
        nb_model.fit(X_train_tfidf, y_train)
        y_pred_nb = nb_model.predict(X_test_tfidf)

        # **📌 计算 Naive Bayes 评估指标**
        conf_mat_nb, report_nb, accuracy_nb = compute_metrics(y_test, y_pred_nb, unique_labels, index_to_label)
        nb_accuracies.append(accuracy_nb)
        nb_conf_mat += conf_mat_nb

        # **📌 训练 kNN**
        knn_model = KNeighborsClassifier(n_neighbors=5, metric="cosine")  # 使用余弦距离
        knn_model.fit(X_train_w2v, y_train)
        y_pred_knn = knn_model.predict(X_test_w2v)

        # **📌 计算 kNN 评估指标**
        conf_mat_knn, report_knn, accuracy_knn = compute_metrics(y_test, y_pred_knn, unique_labels, index_to_label)
        knn_accuracies.append(accuracy_knn)
        knn_conf_mat += conf_mat_knn

        print(f"✅ Naive Bayes Accuracy: {accuracy_nb:.4f}, kNN Accuracy: {accuracy_knn:.4f}\n")

    # **📌 打印最终结果**
    print("\n📌 Final Classification Report (Naive Bayes):")
    print_classification_results(nb_conf_mat, nb_accuracies, unique_labels)

    print("\n📌 Final Classification Report (kNN):")
    print_classification_results(knn_conf_mat, knn_accuracies, unique_labels)

    # **📌 画图**
    plot_cross_validation(nb_accuracies, title="Naive Bayes 10-Fold Accuracy")
    plot_cross_validation(knn_accuracies, title="kNN 10-Fold Accuracy")
    plot_confusion_matrix(nb_conf_mat, unique_labels, title="Naive Bayes Confusion Matrix")
    plot_confusion_matrix(knn_conf_mat, unique_labels, title="kNN Confusion Matrix")

if __name__ == "__main__":
    main()
