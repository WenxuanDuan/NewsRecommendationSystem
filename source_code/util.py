import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 📌 分类评估

def compute_metrics(y_true, y_pred, class_labels, index_to_label):
    y_true_labels = [index_to_label[idx] for idx in y_true]
    y_pred_labels = [index_to_label[idx] for idx in y_pred]

    conf_mat = confusion_matrix(y_true_labels, y_pred_labels, labels=class_labels)
    report = classification_report(y_true_labels, y_pred_labels, labels=class_labels, output_dict=True)
    accuracy = accuracy_score(y_true_labels, y_pred_labels)

    return conf_mat, report, accuracy

# 📌 打印最终报告表格

def print_classification_results(conf_mat, fold_accuracies, class_labels):
    report = classification_report(
        np.repeat(class_labels, conf_mat.sum(axis=1)),
        np.repeat(class_labels, conf_mat.sum(axis=0)),
        labels=class_labels,
        output_dict=True
    )

    table = [["Category", "Precision", "Recall", "F1-Score", "Accuracy"]]
    for label in class_labels:
        precision = report[label]["precision"]
        recall = report[label]["recall"]
        f1_score = report[label]["f1-score"]
        accuracy = conf_mat[class_labels.index(label), class_labels.index(label)] / np.sum(
            conf_mat[class_labels.index(label), :])
        table.append([label, f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}", f"{accuracy:.4f}"])

    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean([report[label]["precision"] for label in class_labels])
    avg_recall = np.mean([report[label]["recall"] for label in class_labels])
    avg_f1_score = np.mean([report[label]["f1-score"] for label in class_labels])

    table.append(["Overall", f"{avg_precision:.4f}", f"{avg_recall:.4f}", f"{avg_f1_score:.4f}", f"{avg_accuracy:.4f}"])

    col_widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    dash_line = " | ".join(["-" * w for w in col_widths])

    print("\n📌 Final Classification Report (Averaged Over 10 Folds):")
    print(dash_line)
    for i, row in enumerate(table):
        print(" | ".join(f"{cell:>{col_widths[j]}}" for j, cell in enumerate(row)))
        if i == 0:
            print(dash_line)
    print(dash_line)

# 📌 可视化混淆矩阵

def plot_confusion_matrix(conf_mat, class_labels, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# 📌 可视化交叉验证准确率

def plot_cross_validation(fold_accuracies, title="10-Fold Cross Validation Accuracy"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, marker="o", linestyle="-", color="b")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.ylim([min(fold_accuracies) - 0.02, 1.0])
    plt.title(title)
    plt.grid()
    plt.show()
