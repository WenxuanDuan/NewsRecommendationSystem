# 📰 News Recommendation System

A content-based news recommendation system built with **Streamlit**.  
It allows users to discover and explore articles through an elegant web interface — with real-time recommendations, personalized reading, and no-repeat logic.

## 🔍 Features

- 🎯 **Random Exploration**  
  On load, users are shown 5 randomly selected articles (title + preview). Click to read or refresh for new content.

- 📖 **Interactive Reading**  
  Click any article card to open and read the full content instantly on the same page.

- 🤖 **Content-Based Recommendations**  
  After reading, 5 similar articles are recommended using a **KNN + Word2Vec** model.

- 🚫 **No Repeats**  
  Already-read articles will not be shown again in homepage or recommendations.

- 🔄 **Re-Randomization Option**  
  Refresh articles anytime to explore other content categories for diversity.

- 💡 **Efficient Local Inference**  
  Word2Vec vectors are computed locally; no external API needed.

## 🧠 How It Works

1. **Preprocessing**:
   - Articles are cleaned using NLTK (tokenization, stopword removal, lemmatization).
   - Word2Vec embeddings are averaged to form document vectors.
![datapreprocessing.png](../Figures/datapreprocessing.png)

2. **Classification (optional)**:
   - A full 10-fold cross-validation pipeline evaluates different models like XGBoost, kNN, Neural Net, and SBERT.

![ClassificationModels.png](../Figures/ClassificationModels.png)
3. **Recommendation**:
   - Based on the selected article, its vector is compared (cosine similarity) with others.
   - The system uses `sklearn.neighbors.NearestNeighbors` for fast nearest-neighbor search.
![UserInteraction.png](../Figures/UserInteraction.png)
   - 
4. **Frontend**:
   - Built with **Streamlit** and designed for interactivity.
   - Uses session state and query parameters to support seamless navigation between views.

## 📸 UI Preview

| Homepage | Read + Recommend |
|----------|-----------------|
|![homepage.png](../Figures/homepage.png) | ![recommendation.png](../Figures/recommendation.png) |

## 🚀 Get Started

```bash
git clone https://github.com/your-username/news-recommendation-system.git
cd news-recommendation-system
pip install -r requirements.txt
streamlit run source_code/app.py
```

## 🧾 Requirements

- Python 3.10+
- streamlit
- numpy
- scikit-learn
- sentence-transformers
- nltk
- gensim
- pandas

(You can export exact dependencies using Poetry or Pip as needed)

## 🌐 Deployment

This project is ready for deployment to [Streamlit Cloud](https://streamlit.io/cloud).  
Just push your repo and make sure the following are present:

- `requirements.txt`
- `source_code/app.py`
- `data/` folder with:
  - `article_vectors.npy`
  - `article_titles.json`
  - `article_labels.json`
  - `documents.pkl`

## 📁 Project Structure

```
news-recommendation-system/
├── data/
│   ├── article_vectors.npy
│   ├── article_titles.json
│   ├── article_labels.json
│   └── documents.pkl
├── source_code/
│   ├── app.py
│   ├── document.py
│   ├── recommendation.py
│   ├── recommendation_utils.py
│   ├── classification.py
│   └── run_classifier.py
├── requirements.txt
└── README.md
```

## 💬 Acknowledgements

- BBC News Dataset (for demo purposes)
- Gensim Word2Vec (Google News 300-d pretrained vectors)
- Huggingface SBERT (`paraphrase-MiniLM-L6-v2`)

---

Feel free to star the repo and try the demo online!
![streamlit_qr.png](../Figures/streamlit_qr.png)