import streamlit as st
import random
from recommendation_utils import (
    load_article_data,
    build_knn_model,
    get_summary,
    recommend_articles
)

# === 加载数据（缓存）
@st.cache_data(show_spinner=False)
def load_all():
    vectors, titles, labels, documents = load_article_data()
    knn_model = build_knn_model(vectors)
    return vectors, titles, labels, documents, knn_model

vectors, titles, labels, documents, knn_model = load_all()

# === 页面状态变量 ===
if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_index" not in st.session_state:
    st.session_state.current_index = None
if "read_indices" not in st.session_state:
    st.session_state.read_indices = set()
if "candidate_indices" not in st.session_state:
    st.session_state.candidate_indices = random.sample(range(len(titles)), 5)


# === 统一卡片样式 ===
import streamlit as st

def article_card(idx):
    from streamlit import session_state as state

    summary = get_summary(documents[idx])
    label = labels[idx]
    title = titles[idx]

    # 👉 样式 + 内容 HTML
    html = f"""
    <div style="border: 1px solid #ccc; border-radius: 12px; padding: 1rem;
                margin-bottom: 1rem; background-color: #fdfdfd;
                text-align: left; cursor: pointer;"
         onclick="document.getElementById('card_btn_{idx}').click();">
        <h4>{title}</h4>
        <p style="margin-top: -10px; color: gray;"><b>Category:</b> {label}</p>
        <p>{summary}</p>
    </div>
    """

    # ✅ 点击按钮逻辑
    if st.button(label="", key=f"card_btn_{idx}"):
        state.current_index = idx
        state.page = "read"
        state.read_indices.add(idx)
        st.rerun()

    # ✅ 展示卡片（会响应点击）
    st.markdown(html, unsafe_allow_html=True)


def show_grid(indices):
    for row in range(0, len(indices), 3):
        cols = st.columns(3)
        for i in range(3):
            if row + i < len(indices):
                with cols[i]:
                    article_card(indices[row + i])


# === 首页 ===
def show_home():
    st.title("📰 News Recommendation System")
    st.markdown("Here are 5 articles you might be interested in:")

    show_grid(st.session_state.candidate_indices)

    if st.button("🔄 Refresh"):
        st.session_state.candidate_indices = random.sample(range(len(titles)), 5)
        st.rerun()


# === 阅读 + 推荐页面 ===
def show_article_page():
    idx = st.session_state.current_index
    st.title(titles[idx])
    st.caption(f"**Category:** {labels[idx]}")
    st.write(documents[idx])  # ✅ 正文（不重复标题）

    st.markdown("---")
    st.subheader("📢 You may also like:")

    recommended = recommend_articles(knn_model, idx, st.session_state.read_indices)
    if recommended:
        show_grid(recommended)
    else:
        st.info("No more recommendations available.")

    if st.button("🏠 Back to Homepage"):
        st.session_state.page = "home"
        st.rerun()


# === 页面路由 ===
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "read":
    show_article_page()
