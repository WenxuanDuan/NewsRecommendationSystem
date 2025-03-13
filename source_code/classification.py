import numpy as np
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# **📌 训练 Naive Bayes + TF-IDF**
def train_naive_bayes(X_train_tfidf, y_train):
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

# **📌 训练 kNN + Word2Vec**
def train_knn(X_train_w2v, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
    model.fit(X_train_w2v, y_train)
    return model

# **📌 训练 XGBoost + TF-IDF**
def train_xgboost(X_train_tfidf, y_train, num_classes):
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    return model
