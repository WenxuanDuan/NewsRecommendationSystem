import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# **📌 训练 Naive Bayes + TF-IDF**
def train_naive_bayes_tfidf(X_train_tfidf, y_train):
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

# **📌 训练 kNN + Word2Vec**
def train_knn_w2v(X_train_w2v, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
    model.fit(X_train_w2v, y_train)
    return model

# **📌 训练 XGBoost + TF-IDF**
def train_xgboost_tfidf(X_train_tfidf, y_train, num_classes):
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    return model

# **📌 训练 XGBoost + Word2Vec**
def train_xgboost_w2v(X_train_w2v, y_train, num_classes):
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train_w2v, y_train)
    return model

# **📌 定义一个简单的前馈神经网络**
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)

# **📌 训练 Word2Vec + Neural Network**
def train_neural_net_w2v(X_train, y_train, X_val, num_classes, epochs=20, hidden_dim=128, lr=0.001):
    input_dim = X_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # 返回训练好的模型
    return model
