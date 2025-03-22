import os
import re
import nltk
import numpy as np
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# # 📌 预加载词向量模型和SBERT模型
# print("\n📌 Loading Pre-trained Word2Vec Model...")
# w2v_model = api.load("word2vec-google-news-300")
# sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 📌 文本预处理

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(processed_tokens)

# ⏳ Word2Vec 模型缓存变量
_w2v_model = None

def get_w2v_model():
    global _w2v_model
    if _w2v_model is None:
        print("\n📌 Loading Pre-trained Word2Vec Model...\n")
        _w2v_model = api.load("word2vec-google-news-300")
    return _w2v_model

# 📌 文档转换为 Word2Vec 特征

def document_to_w2v(doc, model=None, vector_size=300):
    if model is None:
        model = get_w2v_model()
    words = doc.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(vector_size)


# ⏳ SBERT 模型缓存变量
_sbert_model = None

def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        print("\n📌 Loading Pre-trained SBERT Model...\n")
        _sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return _sbert_model


# 📌 文档转换为 SBERT 特征

def document_to_sbert(documents):
    model = get_sbert_model()
    embeddings = model.encode(documents, show_progress_bar=True)
    return embeddings

# 📌 加载并处理文档（返回文件名用于推荐系统）
def load_and_preprocess_documents(data_dir):
    documents, labels, filenames = [], [], []

    for topic in os.listdir(data_dir):
        topic_path = os.path.join(data_dir, topic)

        if not os.path.isdir(topic_path) or topic.startswith("."):
            continue

        for filename in os.listdir(topic_path):
            file_path = os.path.join(topic_path, filename)
            if not filename.endswith(".txt"):
                continue

            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read().strip()

            processed_text = preprocess_text(content)

            if processed_text:
                documents.append(processed_text)
                labels.append(topic)
                filenames.append(filename)

    return documents, labels, filenames


# Example usage:
# doc = Document("001.txt")
# print("Title:", doc.title)
# print("Text:", doc.text)
# print("Title Tokens:", doc.title_tokens)
# print("Text Tokens:", doc.text_tokens)
# print("Document Terms:", doc.document_terms())
#
# print("Word2Vec Vector Shape:", doc.vector.shape)
# print("Word2Vec Vector:", doc.vector)