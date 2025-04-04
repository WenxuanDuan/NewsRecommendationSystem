import streamlit as st
import random
from recommendation_utils import (
    load_article_data,
    build_knn_model,
    get_summary,
    recommend_articles
)

# === 缓存数据加载 ===
@st.cache_data(show_spinner=False)
def load_all():
    vectors, titles, labels, documents = load_article_data()
    knn_model = build_knn_model(vectors)
    return vectors, titles, labels, documents, knn_model

vectors, titles, labels, documents, knn_model = load_all()

# === 页面状态初始化 ===
query_params = st.query_params
page = query_params.get("page", "home")
article_id = query_params.get("article", None)
read_indices = st.session_state.get("read_indices", set())
st.session_state.read_indices = read_indices  # 确保写回

# === 卡片组件 ===
def article_card(idx):
    summary = get_summary(documents[idx])
    label = labels[idx]
    title = titles[idx]

    # 用 query_params 控制跳转
    card_url = f"?page=read&article={idx}"
    card_html = f"""
    <a href="{card_url}" style="text-decoration: none; color: inherit;">
    <div style="
        border: 1px solid #ccc;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #fdfdfd;
        text-align: left;
        cursor: pointer;
        height: 100%;
    ">
        <h4>{title}</h4>
        <p style="margin-top: -10px; color: gray;"><b>Category:</b> {label}</p>
        <p>{summary}</p>
    </div>
    </a>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# === 文章卡片网格布局 ===
def show_grid(indices):
    for row in range(0, len(indices), 3):
        cols = st.columns(3)
        for i in range(3):
            if row + i < len(indices):
                with cols[i]:
                    article_card(indices[row + i])

# === 首页界面 ===
def show_home():
    st.title("📰 News Recommendation System")
    st.markdown("Here are 5 articles you might be interested in:")

    if "candidate_indices" not in st.session_state:
        st.session_state.candidate_indices = random.sample(range(len(titles)), 6)

    show_grid(st.session_state.candidate_indices)

    if st.button("🔄 Refresh"):
        st.session_state.candidate_indices = random.sample(range(len(titles)), 6)
        st.experimental_rerun()

# === 阅读文章 + 推荐页面 ===
def show_article_page(idx):
    idx = int(idx)
    st.session_state.read_indices.add(idx)

    st.title(titles[idx])
    st.caption(f"**Category:** {labels[idx]}")
    st.write(documents[idx])

    st.markdown("---")
    st.subheader("📢 You may also like:")

    recommended = recommend_articles(knn_model, idx, st.session_state.read_indices)
    if recommended:
        show_grid(recommended)
    else:
        st.info("No more recommendations available.")

    if st.button("🏠 Back to Homepage"):
        st.query_params.clear()
        st.experimental_rerun()

# === 路由控制 ===
if page == "read" and article_id is not None:
    show_article_page(article_id)
else:
    show_home()
