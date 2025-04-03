import os
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from document import load_and_preprocess_documents  # 确保模块路径正确

# 📁 设置路径
data_dir = "../data/bbc"           # ← 替换为你的实际数据路径
output_dir = "wordcloud_outputs" # ← 输出图片目录

# 🔠 自定义停用词（建议添加更多）
custom_stopwords = {"said", "mr", "one", "two", "also", "would", "could", "us", "like", "u"}
final_stopwords = STOPWORDS.union(custom_stopwords)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 🔄 载入文档数据
documents, labels, filenames = load_and_preprocess_documents(data_dir)

# 📊 聚合文本
category_texts = defaultdict(str)
for text, label in zip(documents, labels):
    category_texts[label] += " " + text

# ☁️ 生成词云并保存含标题的图像
for category, text in category_texts.items():
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=final_stopwords,
        collocations=False
    ).generate(text)

    # 使用 matplotlib 添加标题后保存
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud – {category.capitalize()}", fontsize=20)

    save_path = os.path.join(output_dir, f"wordcloud_{category.lower()}.png")
    plt.savefig(save_path, bbox_inches='tight')  # ← 保存带标题的完整图像
    print(f"✅ Saved: {save_path}")
    plt.close()  # 不显示图像，节省内存




