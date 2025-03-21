import numpy as np
import torch
from sklearn.model_selection import KFold
from util import compute_metrics
from classification import (
    train_naive_bayes_tfidf,
    train_knn_w2v,
    train_xgboost_tfidf,
    train_xgboost_w2v,
    train_neural_net_w2v,
    train_neural_net_sbert,
    train_xgboost_sbert
)

def run_all_classifiers(X_tfidf, X_w2v, X_sbert, y, unique_labels, index_to_label):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    results = {
        "nb": {"accuracies": [], "conf_mat": np.zeros((len(unique_labels), len(unique_labels)), dtype=int)},
        "knn": {"accuracies": [], "conf_mat": np.zeros((len(unique_labels), len(unique_labels)), dtype=int)},
        "xgb_tfidf": {"accuracies": [], "conf_mat": np.zeros((len(unique_labels), len(unique_labels)), dtype=int)},
        "xgb_w2v": {"accuracies": [], "conf_mat": np.zeros((len(unique_labels), len(unique_labels)), dtype=int)},
        "nn_w2v": {"accuracies": [], "conf_mat": np.zeros((len(unique_labels), len(unique_labels)), dtype=int)},
        "xgb_sbert": {"accuracies": [], "conf_mat": np.zeros((len(unique_labels), len(unique_labels)), dtype=int)},
        "nn_sbert": {"accuracies": [], "conf_mat": np.zeros((len(unique_labels), len(unique_labels)), dtype=int)}
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tfidf), 1):
        print(f"\n🔹 Fold {fold}/10")

        y_train, y_test = y[train_idx], y[test_idx]

        # === Naive Bayes + TF-IDF ===
        X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
        model = train_naive_bayes_tfidf(X_train, y_train)
        y_pred = model.predict(X_test)
        conf_mat, _, acc = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        results["nb"]["accuracies"].append(acc)
        results["nb"]["conf_mat"] += conf_mat

        # === kNN + Word2Vec ===
        X_train, X_test = X_w2v[train_idx], X_w2v[test_idx]
        model = train_knn_w2v(X_train, y_train)
        y_pred = model.predict(X_test)
        conf_mat, _, acc = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        results["knn"]["accuracies"].append(acc)
        results["knn"]["conf_mat"] += conf_mat

        # === XGBoost + TF-IDF ===
        X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
        model = train_xgboost_tfidf(X_train, y_train, num_classes=len(unique_labels))
        y_pred = model.predict(X_test)
        conf_mat, _, acc = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        results["xgb_tfidf"]["accuracies"].append(acc)
        results["xgb_tfidf"]["conf_mat"] += conf_mat

        # === XGBoost + Word2Vec ===
        X_train, X_test = X_w2v[train_idx], X_w2v[test_idx]
        model = train_xgboost_w2v(X_train, y_train, num_classes=len(unique_labels))
        y_pred = model.predict(X_test)
        conf_mat, _, acc = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        results["xgb_w2v"]["accuracies"].append(acc)
        results["xgb_w2v"]["conf_mat"] += conf_mat
        results["xgb_w2v"]["model"] = model

        # === NN + Word2Vec ===
        model = train_neural_net_w2v(X_train, y_train, X_test, num_classes=len(unique_labels))
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            outputs = model(X_test_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        conf_mat, _, acc = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        results["nn_w2v"]["accuracies"].append(acc)
        results["nn_w2v"]["conf_mat"] += conf_mat

        # === XGBoost + SBERT ===
        X_train, X_test = X_sbert[train_idx], X_sbert[test_idx]
        model = train_xgboost_sbert(X_train, y_train, num_classes=len(unique_labels))
        y_pred = model.predict(X_test)
        conf_mat, _, acc = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        results["xgb_sbert"]["accuracies"].append(acc)
        results["xgb_sbert"]["conf_mat"] += conf_mat

        # === NN + SBERT ===
        model = train_neural_net_sbert(X_train, y_train, X_test, num_classes=len(unique_labels))
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            outputs = model(X_test_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        conf_mat, _, acc = compute_metrics(y_test, y_pred, unique_labels, index_to_label)
        results["nn_sbert"]["accuracies"].append(acc)
        results["nn_sbert"]["conf_mat"] += conf_mat

    return results
