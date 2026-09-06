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
    ],

    "Rankings": [
        st.Page(
            "pages/sec_ranking.py",
            title="SEC Weekly Rankings",
            icon="🏆",
        ),
    ],

    # "Monte-Carlo Predictions": [
        # st.Page(
            # "pages/monte_carlo_app.py",
            # title="Simplier Predictor",
            # icon="🎯",
        # ),
    # ],
}

page = st.navigation(
    pages,
    position="top",
)

page.run()