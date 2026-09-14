"""
app.py

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from fantasy_transformer import FantasyScoring, TrainConfig, train_and_rank


st.set_page_config(
    page_title="NFL Fantasy Transformer",
    page_icon="🏈",
    layout="wide",
)

st.title("NFL Fantasy Transformer")
st.caption(
    "Train a SportsDataverse-backed Transformer and build a draft board. "
    "Veterans use NFL weekly-history sequences; rookies use draft/combine priors."
)

with st.sidebar:
    st.header("Training settings")

    current_year = datetime.now().year
    target_season = st.number_input(
        "Draft / target season",
        min_value=2012,
        max_value=current_year + 1,
        value=current_year,
        step=1,
    )

    start_season = st.number_input(
        "Earliest data season",
        min_value=2000,
        max_value=int(target_season) - 3,
        value=min(2012, int(target_season) - 3),
        step=1,
    )

    scoring_name = st.selectbox(
        "Reception scoring",
        ["PPR", "Half PPR", "Standard"],
        index=0,
    )
    reception_points = {
        "PPR": 1.0,
        "Half PPR": 0.5,
        "Standard": 0.0,
    }[scoring_name]

    sequence_length = st.slider(
        "Game-history length",
        min_value=8,
        max_value=68,
        value=34,
        step=2,
        help="34 is roughly two regular seasons of weekly history.",
    )

    epochs = st.slider(
        "Training epochs",
        min_value=2,
        max_value=50,
        value=12,
        step=1,
    )

    batch_size = st.select_slider(
        "Batch size",
        options=[32, 64, 128, 256],
        value=128,
    )

    st.divider()
    st.caption(
        "The first run downloads/caches SportsDataverse data. "
        "Training is intentionally started only by the button below."
    )

train_clicked = st.button(
    "Train model & build draft rankings",
    type="primary",
    use_container_width=True,
)

if train_clicked:
    config = TrainConfig(
        start_season=int(start_season),
        sequence_length=int(sequence_length),
        epochs=int(epochs),
        batch_size=int(batch_size),
    )
    scoring = FantasyScoring(reception=float(reception_points))

    status = st.status("Starting training...", expanded=True)
    message_slot = status.empty()

    def update_status(message: str) -> None:
        message_slot.write(message)

    try:
        result = train_and_rank(
            target_season=int(target_season),
            config=config,
            scoring=scoring,
            output_dir="models",
            source="nflverse",
            progress=update_status,
        )
        st.session_state["fantasy_result"] = result
        st.session_state["fantasy_target_season"] = int(target_season)
        st.session_state["fantasy_scoring_name"] = scoring_name
        status.update(label="Training complete", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Training failed", state="error", expanded=True)
        st.exception(exc)

if "fantasy_result" in st.session_state:
    result = st.session_state["fantasy_result"]
    rankings = result["rankings"]
    metrics = result["metrics"]
    history = result["history"]

    st.subheader(
        f"{st.session_state['fantasy_target_season']} "
        f"{st.session_state['fantasy_scoring_name']} draft board"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validation season", int(metrics["validation_season"]))
    m2.metric("Validation MAE", f"{metrics['mae']:.1f} pts")
    m3.metric("Veterans ranked", int(metrics["ranked_veterans"]))
    m4.metric("Rookies ranked", int(metrics["ranked_rookies"]))

    st.caption(
        "MAE is next-season total fantasy-point error on the held-out most "
        "recent historical season. Treat these rankings as a model baseline, "
        "not as a finished draft product."
    )

    display_cols = [
        "overall_rank",
        "player",
        "pos_rank",
        "projected_points",
        "floor",
        "ceiling",
        "vorp",
        "rookie",
        "model",
    ]
    available_cols = [c for c in display_cols if c in rankings.columns]

    st.dataframe(
        rankings[available_cols].round(
            {
                "projected_points": 1,
                "floor": 1,
                "ceiling": 1,
                "vorp": 1,
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=680,
    )

    csv = rankings.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download draft rankings CSV",
        data=csv,
        file_name=(
            f"fantasy_rankings_"
            f"{st.session_state['fantasy_target_season']}.csv"
        ),
        mime="text/csv",
    )

    with st.expander("Training history"):
        st.line_chart(
            history.set_index("epoch")[["val_mae", "val_rmse"]]
        )
        st.dataframe(history, use_container_width=True, hide_index=True)

    with st.expander("Saved model files"):
        for label, path in result["paths"].items():
            st.code(f"{label}: {path}")
else:
    st.info(
        "Choose the training settings in the sidebar and click "
        "**Train model & build draft rankings**."
    )

st.divider()
st.markdown(
    """
### What this first version is doing

**Veterans:** weekly NFL stats → Transformer → projected next-season fantasy points.

**Rookies:** draft capital + combine + size + position → historical rookie model
→ projected rookie-season fantasy points.

**Draft board:** projected points → position-specific replacement baseline → VORP.

The next major upgrade should add college production to the rookie model and
then backtest the rankings season-by-season before connecting Sleeper.
"""
)
