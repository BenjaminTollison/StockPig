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

    "Predictions": [
        st.Page(
            "pages/predictor_model.py",
            title="Score Predictor",
            icon="🎯",
        ),
        st.Page(
            "pages/monte_carlo_app.py",
            title="Simplier Predictor",
            icon="🎯",
        ),
        st.Page(
            "pages/pickem_optimizer.py",
            title="Your Chances of Beating the League",
            icon="🎯",
        ),
    ],

    "Rankings": [
        st.Page(
            "pages/sec_ranking.py",
            title="SEC Weekly Rankings",
            icon="🏆",
        ),
    ],
}

page = st.navigation(
    pages,
    position="top",
)

page.run()