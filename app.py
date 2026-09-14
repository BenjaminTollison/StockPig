import streamlit as st

pages = {
    "Home": [
        st.Page(
            "pages/home.py",
            title="Home",
            icon="🏠",
            default=True,
        ),
    ],

    "SEC Pickem Tools": [
        st.Page(
            "pages/predictor_model.py",
            title="Score Predictor",
            icon="🎯",
        ),
        st.Page(
            "pages/score_predictor_gpu.py",
            title="Score Predictor GPU",
            icon="🎯",
        ),
        st.Page(
            "pages/monte_carlo_app.py",
            title="Monte-Carlo Predictor",
            icon="🎯",
        ),
        st.Page(
            "pages/pickem_optimizer.py",
            title="How to win at Pickems",
            icon="🎯",
        ),
        st.Page(
            "pages/pickem_optimizer_gpu.py",
            title="How to win at Pickems GPU",
            icon="🎯",
        ),
        st.Page(
            "pages/sec_ranking.py",
            title="SEC Weekly Rankings",
            icon="🏆",
        ),
    ],
    "NFL Fantasy": [
        st.Page(
            "fantasy_football/transformer_test.py",
            title="Transformer Test",
            icon="🎯",
        ),
    ],
}

page = st.navigation(
    pages,
    position="top",
)

page.run()