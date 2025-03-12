import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def compute_metrics(y_true, y_pred, class_labels, index_to_label):
    """
    计算混淆矩阵、分类报告和准确率。

    :param y_true: 真实标签（数值）
    :param y_pred: 预测标签（数值）
    :param class_labels: 类别名称（字符串列表）
    :param index_to_label: 从数值索引到类别名称的映射
    :return: 混淆矩阵、分类报告（字典）、准确率
    """
    # **将数值索引转换回类别名**
    y_true_labels = [index_to_label[idx] for idx in y_true]
    y_pred_labels = [index_to_label[idx] for idx in y_pred]

    # **计算混淆矩阵**
    conf_mat = confusion_matrix(y_true_labels, y_pred_labels, labels=class_labels)

    # **计算分类报告**
    report = classification_report(y_true_labels, y_pred_labels, labels=class_labels, output_dict=True)

    # **计算准确率**
    accuracy = accuracy_score(y_true_labels, y_pred_labels)

    return conf_mat, report, accuracy


def print_classification_results(conf_mat, fold_accuracies, class_labels):
    """
    以表格形式打印最终的分类结果，包括 precision, recall, f1-score 和 accuracy。

    :param conf_mat: 总混淆矩阵
    :param fold_accuracies: 10-Fold 交叉验证准确率列表
    :param class_labels: 类别名称
    """
    # **计算 Precision, Recall, F1-Score**
    report = classification_report(np.repeat(class_labels, conf_mat.sum(axis=1)),
                                   np.repeat(class_labels, conf_mat.sum(axis=0)),
                                   labels=class_labels, output_dict=True)

    table = [["Category", "Precision", "Recall", "F1-Score", "Accuracy"]]
    for label in class_labels:
        precision = report[label]["precision"]
        recall = report[label]["recall"]
        f1_score = report[label]["f1-score"]
        accuracy = conf_mat[class_labels.index(label), class_labels.index(label)] / np.sum(
            conf_mat[class_labels.index(label), :])
        table.append([label, f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}", f"{accuracy:.4f}"])

    # **计算总体 Accuracy**
    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean([report[label]["precision"] for label in class_labels])
    avg_recall = np.mean([report[label]["recall"] for label in class_labels])
    avg_f1_score = np.mean([report[label]["f1-score"] for label in class_labels])

    table.append(["Overall", f"{avg_precision:.4f}", f"{avg_recall:.4f}", f"{avg_f1_score:.4f}", f"{avg_accuracy:.4f}"])

    # **打印表格**
    col_widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    dash_line = " | ".join(["-" * w for w in col_widths])

    print("\n📌 Final Classification Report (Averaged Over 10 Folds):")
    print(dash_line)
    for i, row in enumerate(table):
        print(" | ".join(f"{cell:>{col_widths[j]}}" for j, cell in enumerate(row)))
        if i == 0:
            print(dash_line)
    print(dash_line)


def plot_confusion_matrix(conf_mat, class_labels, title="Confusion Matrix"):
    """
    绘制混淆矩阵的热力图。

    :param conf_mat: 混淆矩阵
    :param class_labels: 类别名称
    :param title: 图表标题
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def plot_cross_validation(fold_accuracies, title="10-Fold Cross Validation Accuracy"):
    """
    绘制 10-Fold 交叉验证准确率变化折线图。

    :param fold_accuracies: 每个 fold 的准确率列表
    :param title: 图表标题
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, marker="o", linestyle="-", color="b")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.ylim([min(fold_accuracies) - 0.02, 1.0])
    plt.title(title)
    plt.grid()
    plt.show()
