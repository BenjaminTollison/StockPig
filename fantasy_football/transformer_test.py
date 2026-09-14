"""
transformer_test.py

Run with:
    streamlit run fantasy_football/transformer_test.py
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from fantasy_transformer import FantasyScoring, TrainConfig, train_and_rank


def rank_veterans_and_rookies_together(
    result: dict,
    config: TrainConfig,
) -> pd.DataFrame:
    """
    Build one draft board containing veterans and rookies.

    This intentionally recalculates replacement level and VORP *after* both
    groups are combined so a rookie can rank above/below veterans naturally.
    It also supports either of these result shapes:

      result["rankings"]

    or separate frames:

      result["veteran_rankings"]
      result["rookie_rankings"]
    """
    frames: list[pd.DataFrame] = []

    veteran_rankings = result.get("veteran_rankings")
    rookie_rankings = result.get("rookie_rankings")

    if isinstance(veteran_rankings, pd.DataFrame) and not veteran_rankings.empty:
        veteran_rankings = veteran_rankings.copy()
        veteran_rankings["rookie"] = False
        if "model" not in veteran_rankings.columns:
            veteran_rankings["model"] = "transformer"
        frames.append(veteran_rankings)

    if isinstance(rookie_rankings, pd.DataFrame) and not rookie_rankings.empty:
        rookie_rankings = rookie_rankings.copy()
        rookie_rankings["rookie"] = True
        if "model" not in rookie_rankings.columns:
            rookie_rankings["model"] = "rookie_prior"
        frames.append(rookie_rankings)

    # The current fantasy_transformer.train_and_rank() returns a combined
    # rankings frame. Use it when separate frames are not supplied.
    if not frames:
        rankings = result.get("rankings")
        if not isinstance(rankings, pd.DataFrame):
            raise ValueError(
                "Training result did not contain a rankings DataFrame."
            )
        frames.append(rankings.copy())

    board = pd.concat(frames, ignore_index=True, sort=False)

    required = {"player", "position", "projected_points"}
    missing = required.difference(board.columns)
    if missing:
        raise ValueError(
            "Ranking data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    board["position"] = board["position"].astype(str).str.upper()
    board["projected_points"] = pd.to_numeric(
        board["projected_points"], errors="coerce"
    ).fillna(0.0)

    if "rookie" not in board.columns:
        board["rookie"] = False
    board["rookie"] = board["rookie"].fillna(False).astype(bool)

    if "model" not in board.columns:
        board["model"] = np.where(
            board["rookie"], "rookie_prior", "transformer"
        )

    # Avoid duplicate players if the model result happens to contain the same
    # player in more than one returned frame. Prefer the higher projection.
    dedupe_cols = [c for c in ["player_id", "player", "position"] if c in board.columns]
    if dedupe_cols:
        board = (
            board.sort_values("projected_points", ascending=False)
            .drop_duplicates(subset=dedupe_cols, keep="first")
            .reset_index(drop=True)
        )

    replacement_ranks = {
        "QB": config.replacement_rank_qb,
        "RB": config.replacement_rank_rb,
        "WR": config.replacement_rank_wr,
        "TE": config.replacement_rank_te,
    }

    replacement_points: dict[str, float] = {}
    for position, replacement_rank in replacement_ranks.items():
        values = (
            board.loc[board["position"].eq(position), "projected_points"]
            .sort_values(ascending=False)
            .to_numpy()
        )
        if len(values) == 0:
            replacement_points[position] = 0.0
        else:
            idx = min(max(int(replacement_rank) - 1, 0), len(values) - 1)
            replacement_points[position] = float(values[idx])

    board["replacement_points"] = (
        board["position"].map(replacement_points).fillna(0.0)
    )
    board["vorp"] = (
        board["projected_points"] - board["replacement_points"]
    )

    # Preserve model-provided intervals when available. Otherwise derive an
    # approximate 80% interval from the predicted uncertainty.
    if "uncertainty" in board.columns:
        uncertainty = pd.to_numeric(
            board["uncertainty"], errors="coerce"
        ).fillna(0.0)
        if "floor" not in board.columns:
            board["floor"] = np.maximum(
                0.0, board["projected_points"] - 1.28 * uncertainty
            )
        if "ceiling" not in board.columns:
            board["ceiling"] = (
                board["projected_points"] + 1.28 * uncertainty
            )

    # One shared sort means rookies and veterans compete for the same overall
    # spots. VORP is primary; raw projected points break ties.
    board = board.sort_values(
        ["vorp", "projected_points"],
        ascending=[False, False],
    ).reset_index(drop=True)

    board["overall_rank"] = np.arange(1, len(board) + 1)
    board["position_rank"] = board.groupby("position").cumcount() + 1
    board["pos_rank"] = (
        board["position"] + board["position_rank"].astype(str)
    )

    return board


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
        min_value=2016,
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
        max_value=150,
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

# Build the config on every Streamlit rerun so the unified ranking code can
# reuse the same replacement-level settings even when the train button is not
# the event that triggered the rerun.
config = TrainConfig(
    start_season=int(start_season),
    sequence_length=int(sequence_length),
    epochs=int(epochs),
    batch_size=int(batch_size),
)
scoring = FantasyScoring(reception=float(reception_points))

train_clicked = st.button(
    "Train model & build draft rankings",
    type="primary",
    use_container_width=True,
)

if train_clicked:

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
    rankings = rank_veterans_and_rookies_together(
        result=result,
        config=config,
    )
    metrics = result["metrics"]
    history = result["history"]

    st.subheader(
        f"{st.session_state['fantasy_target_season']} "
        f"{st.session_state['fantasy_scoring_name']} draft board"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validation season", int(metrics["validation_season"]))
    m2.metric("Validation MAE", f"{metrics['mae']:.1f} pts")
    veteran_count = int((~rankings["rookie"]).sum())
    rookie_count = int(rankings["rookie"].sum())
    m3.metric("Veterans ranked", veteran_count)
    m4.metric("Rookies ranked", rookie_count)

    veteran_source_season = metrics.get("veteran_source_season")

    if veteran_count == 0:
        st.error(
            "No veteran projections were produced. Open **Projection diagnostics** "
            "below to see where veteran candidates were lost."
        )
    elif (
        veteran_source_season is not None
        and int(veteran_source_season)
        < int(st.session_state["fantasy_target_season"]) - 1
    ):
        st.warning(
            f"The newest veteran stat season available is "
            f"{int(veteran_source_season)}. Rankings for "
            f"{int(st.session_state['fantasy_target_season'])} therefore use an "
            "older-than-expected veteran history."
        )

    with st.expander("Projection diagnostics"):
        diagnostic_keys = [
            "veteran_source_season",
            "veteran_candidate_count",
            "veteran_samples_built",
            "veteran_metadata_matches",
            "veteran_metadata_misses",
            "ranked_veterans",
            "ranked_rookies",
            "rookie_training_samples",
        ]
        st.json(
            {
                key: metrics.get(key)
                for key in diagnostic_keys
                if key in metrics
            }
        )

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

**Draft board:** veteran and rookie projections are merged first, then one shared
position-specific replacement baseline and VORP ranking is calculated.

The next major upgrade should add college production to the rookie model and
then backtest the rankings season-by-season before connecting Sleeper.
"""
)
