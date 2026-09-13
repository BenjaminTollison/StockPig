import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import torch

from scipy.stats import norm

import sportsdataverse as sdv

from sportsdataverse.cfb.cfb_ratings import cfb_ratings
from sportsdataverse.cfb.cfb_game_predict import cfb_predict_games

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

SEASON = 2026

st.set_page_config(
    page_title="SEC Monte-Carlo Score Predictor",
    page_icon="🏈",
    layout="wide",
)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def load_schedule(season: int):
    """
    Load current + previous season.

    Previous season is useful for estimating uncertainty
    and early-season fallback information.
    """

    schedule = sdv.cfb.load_cfb_schedule(
        seasons=[season - 2, season],
        return_as_pandas=True,
    )

    schedule["start_date"] = pd.to_datetime(
        schedule["start_date"],
        utc=True,
        errors="coerce",
    )

    return schedule


@st.cache_resource(ttl=3600)
def load_pbp(season: int):
    """
    Keep this as Polars because SportsDataverse's
    efficiency_ratings() expects a Polars DataFrame.
    """

    return sdv.cfb.load_cfb_pbp(
        seasons=[season - 2, season]
    )


# ---------------------------------------------------------
# CURRENT WEEK
# ---------------------------------------------------------

def get_default_week(schedule: pd.DataFrame, season: int) -> int:

    games = schedule[
        (schedule["season"] == season)
        & (schedule["season_type_id"] == 2)
    ].copy()

    now = pd.Timestamp.now(tz="UTC")

    weeks = (
        games.groupby("week")["start_date"]
        .agg(["min", "max"])
        .dropna()
    )

    def distance_from_week(row):

        start = row["min"] - pd.Timedelta(days=2)
        end = row["max"] + pd.Timedelta(days=2)

        if start <= now <= end:
            return 0

        return min(
            abs((now - start).total_seconds()),
            abs((now - end).total_seconds()),
        )

    weeks["distance"] = weeks.apply(
        distance_from_week,
        axis=1,
    )

    return int(weeks["distance"].idxmin())


# ---------------------------------------------------------
# SEC GAMES
# ---------------------------------------------------------

def get_sec_games(
    schedule: pd.DataFrame,
    season: int,
    week,
):

    games = schedule[
        (schedule["season"] == season)
        & (schedule["season_type_id"] == 2)
        & (schedule["week"] == week)
    ].copy()

    # Keep a game whenever EITHER team is an SEC team.
    # This includes SEC vs nonconference games.
    games = games[
        (games["home_conference"] == "SEC")
        | (games["away_conference"] == "SEC")
    ]

    return games


# ---------------------------------------------------------
# RATINGS
# ---------------------------------------------------------

def make_ratings(
    schedule: pd.DataFrame,
    season: int,
    week,
):
    """
    Build opponent-adjusted ratings using only games that occurred
    before the selected week.

    Pools the previous season with the current season to provide
    more information early in the year.
    """

    # Get all games from the week we're trying to predict.
    week_games = schedule[
        (schedule["season"] == season)
        & (schedule["week"] == week)
    ].copy()

    if week_games.empty:
        raise ValueError(
            f"No games found for season {season}, week {week}"
        )

    # SportsDataverse's cfb_ratings() uses an as_of_date boundary:
    # only games with date < as_of_date are included.
    #
    # By choosing the earliest kickoff date of the selected week,
    # NO games from that week leak into our ratings.
    week_games["start_date"] = pd.to_datetime(
        week_games["start_date"],
        utc=True,
        errors="coerce",
    )

    first_game_date = week_games["start_date"].min()

    if pd.isna(first_game_date):
        raise ValueError(
            f"Could not determine start date for season {season}, week {week}"
        )

    as_of_date = first_game_date.date()

    ratings = cfb_ratings(
        seasons=[season - 1, season],
        as_of_date=as_of_date,

        # Important for SEC schedules because many early-season
        # opponents are FCS schools.
        fbs_only=False,

        return_as_pandas=False,
    )

    return ratings


# ---------------------------------------------------------
# PREPARE GAMES FOR SPORTSDATAVERSE MODEL
# ---------------------------------------------------------

def prepare_games(
    sec_games: pd.DataFrame,
    ratings: pl.DataFrame,
):

    games = pl.DataFrame({
        "game_id": sec_games["game_id"].astype(int),
        "home_team_id": sec_games["home_id"].astype(str),
        "away_team_id": sec_games["away_id"].astype(str),
        "neutral_site": sec_games["neutral_site"].fillna(False),
    })

    # Make sure every opponent exists in ratings.
    # This matters especially for SEC vs FCS games.

    required = set(
        games["home_team_id"].to_list()
        + games["away_team_id"].to_list()
    )

    existing = set(
        ratings["team_id"].cast(pl.String).to_list()
    )

    missing = required - existing

    if missing:

        league_pace = ratings["off_pace"].mean()

        fallback = pl.DataFrame({
            "team_id": list(missing),
            "adj_off_epa": [0.0] * len(missing),
            "adj_def_epa": [0.0] * len(missing),
            "adj_net": [0.0] * len(missing),
            "games": [0] * len(missing),
            "off_pace": [league_pace] * len(missing),
        })

        ratings = pl.concat(
            [ratings, fallback],
            how="diagonal_relaxed",
        )

    ratings = ratings.with_columns(
        pl.col("team_id").cast(pl.String)
    )

    return games, ratings


# ---------------------------------------------------------
# SCORE PREDICTIONS
# ---------------------------------------------------------

def predict_games(
    sec_games: pd.DataFrame,
    ratings: pl.DataFrame,
):

    games, ratings = prepare_games(
        sec_games,
        ratings,
    )

    predictions = cfb_predict_games(
        games,
        ratings,
        return_as_pandas=True,
    )

    predictions["game_id"] = (
        predictions["game_id"]
        .astype(int)
    )

    result = sec_games.merge(
        predictions,
        on="game_id",
        how="left",
    )

    # Convert margin + total into team scores.

    result["expected_home_score"] = (
        result["exp_total"]
        + result["exp_margin"]
    ) / 2

    result["expected_away_score"] = (
        result["exp_total"]
        - result["exp_margin"]
    ) / 2

    return result


# ---------------------------------------------------------
# UNCERTAINTY
# ---------------------------------------------------------

def estimate_total_sd(schedule: pd.DataFrame):

    completed = schedule[
        (schedule["completed"] == True)
        & schedule["home_points"].notna()
        & schedule["away_points"].notna()
    ].copy()

    totals = (
        completed["home_points"]
        + completed["away_points"]
    )

    return float(totals.std())


def estimate_margin_sd(predictions: pd.DataFrame):

    # SportsDataverse defines:
    #
    # home_win_prob =
    # Phi(exp_margin / margin_sd)
    #
    # therefore:
    #
    # margin_sd =
    # exp_margin / Phi^-1(home_win_prob)

    values = []

    for _, row in predictions.iterrows():

        p = row["home_win_prob"]
        margin = row["exp_margin"]

        if pd.isna(p) or pd.isna(margin):
            continue

        if p <= 0.01 or p >= 0.99:
            continue

        z = norm.ppf(p)

        if abs(z) < 0.05:
            continue

        values.append(
            abs(margin / z)
        )

    if values:
        return float(np.median(values))

    # Should rarely happen.
    return 16.0


# ---------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------

def simulate_game(
    row,
    margin_sd,
    total_sd,
    n=30000,
):
    if torch.cuda.is_available():
        n=int(1e6)
    seed = 42
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    simulated_margin = (
        torch.rand(
            n,
            device=DEVICE,
            dtype=torch.float32,
            generator=generator,
        )
        * float(margin_sd) + float(row["exp_margin"])
    )
    simulated_total = (
            torch.randn(
                n,
                device=DEVICE,
                dtype=torch.float32,
                generator=generator,
            )
            * float(total_sd)
            + float(row["exp_total"])
        )

    home = torch.round(
            (simulated_total + simulated_margin) / 2
        )

    away = torch.round(
            (simulated_total - simulated_margin) / 2
        )

    home = torch.clamp(home, min=0).to(torch.int32)
    away = torch.clamp(away, min=0).to(torch.int32)



    return home.cpu().numpy(), away.cpu().numpy()


def score_distribution(scores):

    values, counts = np.unique(
        scores,
        return_counts=True,
    )

    probabilities = counts / len(scores)

    return pd.DataFrame({
        "Score": values,
        "Probability": probabilities,
    })


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

st.title("🏈 SEC Monte-Carlo Weekly Score Predictor")

st.caption(
    "Expected scores and simulated score probabilities "
    "using SportsDataverse CFB efficiency ratings."
)


schedule = load_schedule(SEASON)

pbp = load_pbp(SEASON)

default_week = get_default_week(
    schedule,
    SEASON,
)


available_weeks = sorted(
    schedule[
        schedule["season"] == SEASON
    ]["week"]
    .dropna()
    .astype(int)
    .unique()
)


week = st.sidebar.selectbox(
    "Week",
    available_weeks,
    index=(
        available_weeks.index(default_week)
        if default_week in available_weeks
        else 0
    ),
)

if torch.cuda.is_available():
    st.sidebar.success(
        f"GPU acceleration: {torch.cuda.get_device_name(0)}"
    )
else:
    st.sidebar.warning("GPU unavailable — using CPU")

sec_games = get_sec_games(
    schedule,
    SEASON,
    week,
)


if sec_games.empty:

    st.warning(
        "No SEC games found for this week."
    )

    st.stop()


ratings = make_ratings(
    #pbp,
    schedule,
    SEASON,
    week,
)


predictions = predict_games(
    sec_games,
    ratings,
)


total_sd = estimate_total_sd(schedule)

margin_sd = estimate_margin_sd(predictions)


# ---------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------

summary = predictions[
    [
        "away_team",
        "home_team",
        "expected_away_score",
        "expected_home_score",
        "home_win_prob",
    ]
].copy()


summary["expected_away_score"] = (
    summary["expected_away_score"].round(1)
)

summary["expected_home_score"] = (
    summary["expected_home_score"].round(1)
)

summary["home_win_prob"] = (
    summary["home_win_prob"] * 100
).round(1)


summary.columns = [
    "Away",
    "Home",
    "Away Expected",
    "Home Expected",
    "Home Win %",
]


st.subheader(f"SEC Week {week}")

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# GAME SELECTOR
# ---------------------------------------------------------

game_labels = {
    row["game_id"]:
        f'{row["away_team"]} @ {row["home_team"]}'
    for _, row in predictions.iterrows()
}


selected_game = st.selectbox(
    "Select matchup",
    list(game_labels.keys()),
    format_func=lambda x: game_labels[x],
)


game = predictions[
    predictions["game_id"] == selected_game
].iloc[0]


home_scores, away_scores = simulate_game(
    game,
    margin_sd,
    total_sd,
)


# ---------------------------------------------------------
# GAME HEADER
# ---------------------------------------------------------

st.header(
    f'{game["away_team"]} @ {game["home_team"]}'
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        f'{game["away_team"]} expected',
        f'{game["expected_away_score"]:.1f}',
    )


with col2:

    st.metric(
        f'{game["home_team"]} expected',
        f'{game["expected_home_score"]:.1f}',
    )


with col3:

    st.metric(
        f'{game["home_team"]} win probability',
        f'{game["home_win_prob"] * 100:.1f}%',
    )


# ---------------------------------------------------------
# DISTRIBUTIONS
# ---------------------------------------------------------

away_distribution = score_distribution(
    away_scores
)

home_distribution = score_distribution(
    home_scores
)


left, right = st.columns(2)


with left:

    st.subheader(
        game["away_team"]
    )

    st.bar_chart(
        away_distribution,
        x="Score",
        y="Probability",
    )

    st.write(
        "80% score range:",
        f"{np.percentile(away_scores, 10):.0f}"
        " – "
        f"{np.percentile(away_scores, 90):.0f}",
    )


with right:

    st.subheader(
        game["home_team"]
    )

    st.bar_chart(
        home_distribution,
        x="Score",
        y="Probability",
    )

    st.write(
        "80% score range:",
        f"{np.percentile(home_scores, 10):.0f}"
        " – "
        f"{np.percentile(home_scores, 90):.0f}",
    )


# ---------------------------------------------------------
# MOST LIKELY EXACT SCORES
# ---------------------------------------------------------

st.subheader("Most likely exact scores")


away_top = (
    away_distribution
    .sort_values(
        "Probability",
        ascending=False,
    )
    .head(10)
    .copy()
)

home_top = (
    home_distribution
    .sort_values(
        "Probability",
        ascending=False,
    )
    .head(10)
    .copy()
)


away_top["Probability"] *= 100
home_top["Probability"] *= 100


left, right = st.columns(2)


with left:

    st.write(game["away_team"])

    st.dataframe(
        away_top,
        hide_index=True,
    )


with right:

    st.write(game["home_team"])

    st.dataframe(
        home_top,
        hide_index=True,
    )