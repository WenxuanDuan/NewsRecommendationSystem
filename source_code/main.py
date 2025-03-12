import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from document import load_and_preprocess_documents
from util import compute_metrics, plot_confusion_matrix, plot_cross_validation, print_classification_results

# **📌 主函数**
def main():
    print("📌 Loading and Preprocessing Dataset...\n")
    data_dir = "../data/bbc/"
    documents, labels = load_and_preprocess_documents(data_dir)

    # **📌 处理 labels：字符串转数值索引**
    unique_labels = sorted(set(labels))  # 获取所有类别 ['business', 'entertainment', 'politics', 'sport', 'tech']
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}  # 反向映射

    # **📌 转换 labels 为数值**
    y = np.array([label_to_index[label] for label in labels])

    # **📌 计算 TF-IDF 特征**
    print("\n📌 Training TF-IDF model...\n")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # **📌 10-Fold 交叉验证**
    print("\n📌 Running 10-Fold Cross Validation...\n")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []
    total_conf_mat = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"🔹 Fold {fold}/10")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # **📌 训练 Naive Bayes**
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # **📌 计算评估指标**
        conf_mat, report, accuracy = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        fold_accuracies.append(accuracy)
        total_conf_mat += conf_mat

        print(f"✅ Accuracy: {accuracy:.4f}\n")

    # **📌 计算最终平均指标**
    print("\n📌 Final Classification Report (Averaged Over 10 Folds):")
    print_classification_results(total_conf_mat, fold_accuracies, unique_labels)

    # **📌 可视化**
    plot_cross_validation(fold_accuracies)
    plot_confusion_matrix(total_conf_mat, unique_labels, title="Total Confusion Matrix (All Folds Combined)")

if __name__ == "__main__":
    main()
