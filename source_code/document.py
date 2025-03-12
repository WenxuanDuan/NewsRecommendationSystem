import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# # 下载 NLTK 资源（仅需运行一次）
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

# **定义数据预处理函数**
def preprocess_text(text):
    """对文本进行分词、去停用词、词形还原"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # 1️⃣ **转换为小写**
    text = text.lower()

    # 2️⃣ **移除特殊字符和数字**
    text = re.sub(r"[^a-z\s]", "", text)

    # 3️⃣ **分词**
    tokens = word_tokenize(text)

    # 4️⃣ **去除停用词 & 词形还原**
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return " ".join(processed_tokens)

# **加载并处理文档**
def load_and_preprocess_documents(data_dir):
    """
    读取数据集文件夹，预处理文本，并返回 (文本列表, 类别标签列表)
    """
    documents, labels = [], []

    for topic in os.listdir(data_dir):
        topic_path = os.path.join(data_dir, topic)

        # **忽略 `.DS_Store` 或非目录项**
        if not os.path.isdir(topic_path) or topic.startswith("."):
            continue

        for filename in os.listdir(topic_path):
            file_path = os.path.join(topic_path, filename)

            # **忽略 `.DS_Store` 等非文本文件**
            if not filename.endswith(".txt"):
                continue

            # **读取文件内容**
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read().strip()

            # **预处理文本**
            processed_text = preprocess_text(content)

            if processed_text:  # 过滤空文本
                documents.append(processed_text)
                labels.append(topic)

    return documents, labels


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