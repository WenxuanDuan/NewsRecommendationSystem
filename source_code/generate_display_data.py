import os
import json
import pickle

data_dir = "../data/bbc/"
titles = []
documents = []

for category in sorted(os.listdir(data_dir)):
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        continue

    for filename in sorted(os.listdir(category_path)):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(category_path, filename)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            documents.append(content)

            # 提取第一段或第一行作为标题
            lines = content.splitlines()
            non_empty = [line for line in lines if line.strip()]
            title = non_empty[0] if non_empty else filename
            titles.append(title)

# 保存标题和原文
os.makedirs("data", exist_ok=True)

with open("data/original_article_titles.json", "w") as f:
    json.dump(titles, f)

with open("data/original_documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Finished: original_article_titles.json and original_documents.pkl saved.")
