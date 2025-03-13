import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from document import load_and_preprocess_documents, document_to_w2v
from util import compute_metrics, plot_confusion_matrix, plot_cross_validation, print_classification_results
from classification import (
    train_naive_bayes_tfidf,
    train_knn_w2v,
    train_xgboost_tfidf,
    train_xgboost_w2v
)

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

    # 🔹 Naive Bayes + TF-IDF
    nb_accuracies, nb_conf_mat = [], np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    # 🔹 kNN + Word2Vec
    knn_accuracies, knn_conf_mat = [], np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    # 🔹 XGBoost + TF-IDF
    xgb_tfidf_accuracies, xgb_tfidf_conf_mat = [], np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    # 🔹 XGBoost + Word2Vec
    xgb_w2v_accuracies, xgb_w2v_conf_mat = [], np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tfidf), 1):
        print(f"🔹 Fold {fold}/10")

        # **📌 划分训练集 & 测试集**
        X_train_tfidf, X_test_tfidf = X_tfidf[train_idx], X_tfidf[test_idx]
        X_train_w2v, X_test_w2v = X_w2v[train_idx], X_w2v[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # **📌 训练 Naive Bayes + TF-IDF**
        nb_model = train_naive_bayes_tfidf(X_train_tfidf, y_train)
        y_pred_nb = nb_model.predict(X_test_tfidf)

        # **📌 计算 Naive Bayes 评估指标**
        conf_mat_nb, report_nb, accuracy_nb = compute_metrics(y_test, y_pred_nb, unique_labels, index_to_label)
        nb_accuracies.append(accuracy_nb)
        nb_conf_mat += conf_mat_nb

        # **📌 训练 kNN + Word2Vec**
        knn_model = train_knn_w2v(X_train_w2v, y_train)
        y_pred_knn = knn_model.predict(X_test_w2v)

        # **📌 计算 kNN 评估指标**
        conf_mat_knn, report_knn, accuracy_knn = compute_metrics(y_test, y_pred_knn, unique_labels, index_to_label)
        knn_accuracies.append(accuracy_knn)
        knn_conf_mat += conf_mat_knn

        # **📌 训练 XGBoost + TF-IDF**
        xgb_tfidf_model = train_xgboost_tfidf(X_train_tfidf, y_train, num_classes=len(unique_labels))
        y_pred_xgb_tfidf = xgb_tfidf_model.predict(X_test_tfidf)

        # **📌 计算 XGBoost + TF-IDF 评估指标**
        conf_mat_xgb_tfidf, report_xgb_tfidf, accuracy_xgb_tfidf = compute_metrics(y_test, y_pred_xgb_tfidf, unique_labels, index_to_label)
        xgb_tfidf_accuracies.append(accuracy_xgb_tfidf)
        xgb_tfidf_conf_mat += conf_mat_xgb_tfidf

        # **📌 训练 XGBoost + Word2Vec**
        xgb_w2v_model = train_xgboost_w2v(X_train_w2v, y_train, num_classes=len(unique_labels))
        y_pred_xgb_w2v = xgb_w2v_model.predict(X_test_w2v)

        # **📌 计算 XGBoost + Word2Vec 评估指标**
        conf_mat_xgb_w2v, report_xgb_w2v, accuracy_xgb_w2v = compute_metrics(y_test, y_pred_xgb_w2v, unique_labels, index_to_label)
        xgb_w2v_accuracies.append(accuracy_xgb_w2v)
        xgb_w2v_conf_mat += conf_mat_xgb_w2v

        print(f"✅ Naive Bayes: {accuracy_nb:.4f}, kNN: {accuracy_knn:.4f}, XGBoost TF-IDF: {accuracy_xgb_tfidf:.4f}, XGBoost W2V: {accuracy_xgb_w2v:.4f}\n")

    # **📌 打印最终结果**
    print("\n📌 Final Classification Report (Naive Bayes + TF-IDF):")
    print_classification_results(nb_conf_mat, nb_accuracies, unique_labels)

    print("\n📌 Final Classification Report (kNN + Word2Vec):")
    print_classification_results(knn_conf_mat, knn_accuracies, unique_labels)

    print("\n📌 Final Classification Report (XGBoost + TF-IDF):")
    print_classification_results(xgb_tfidf_conf_mat, xgb_tfidf_accuracies, unique_labels)

    print("\n📌 Final Classification Report (XGBoost + Word2Vec):")
    print_classification_results(xgb_w2v_conf_mat, xgb_w2v_accuracies, unique_labels)

    # **📌 画图**
    plot_cross_validation(nb_accuracies, title="Naive Bayes + TF-IDF Accuracy")
    plot_cross_validation(knn_accuracies, title="kNN + Word2Vec Accuracy")
    plot_cross_validation(xgb_tfidf_accuracies, title="XGBoost + TF-IDF Accuracy")
    plot_cross_validation(xgb_w2v_accuracies, title="XGBoost + Word2Vec Accuracy")

    plot_confusion_matrix(nb_conf_mat, unique_labels, title="Naive Bayes + TF-IDF Confusion Matrix")
    plot_confusion_matrix(knn_conf_mat, unique_labels, title="kNN + Word2Vec Confusion Matrix")
    plot_confusion_matrix(xgb_tfidf_conf_mat, unique_labels, title="XGBoost + TF-IDF Confusion Matrix")
    plot_confusion_matrix(xgb_w2v_conf_mat, unique_labels, title="XGBoost + Word2Vec Confusion Matrix")

if __name__ == "__main__":
    main()
