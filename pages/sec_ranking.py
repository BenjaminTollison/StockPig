from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

import sportsdataverse as sdv
import stat_processing


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SEASON = datetime.now(timezone.utc).year
DEFAULT_LOOKBACK_GAMES = 10

# Current SEC membership, keyed by ESPN team id.
SEC_TEAMS = {
    333: ("Alabama", "ALA"),
    8: ("Arkansas", "ARK"),
    2: ("Auburn", "AUB"),
    57: ("Florida", "FLA"),
    61: ("Georgia", "UGA"),
    96: ("Kentucky", "UK"),
    99: ("LSU", "LSU"),
    344: ("Mississippi State", "MSST"),
    142: ("Missouri", "MIZ"),
    201: ("Oklahoma", "OU"),
    145: ("Ole Miss", "MISS"),
    2579: ("South Carolina", "SC"),
    2633: ("Tennessee", "TENN"),
    251: ("Texas", "TEX"),
    245: ("Texas A&M", "TA&M"),
    238: ("Vanderbilt", "VAN"),
}

OFFENSE_QUALITY = {
    "off_epa_per_play": "higher",
    "off_pass_epa_per_play": "higher",
    "off_rush_epa_per_play": "higher",
    "off_success_rate": "higher",
    "off_turnover_rate": "lower",
    "off_points_per_game": "higher",
}

# These are useful descriptors, but are not inherently "good" or "bad",
# so they are displayed without contributing to the offense composite.
OFFENSE_STYLE = [
    "off_pass_rate",
    "off_plays_per_game",
]

DEFENSE_QUALITY = {
    "def_epa_allowed_per_play": "lower",
    "def_pass_epa_allowed_per_play": "lower",
    "def_rush_epa_allowed_per_play": "lower",
    "def_success_rate_allowed": "lower",
    "def_sack_rate": "higher",
    "def_turnover_forced_rate": "higher",
    "def_points_allowed_per_game": "lower",
}

LABELS = {
    "off_epa_per_play": "EPA / Play",
    "off_pass_epa_per_play": "Pass EPA / Play",
    "off_rush_epa_per_play": "Rush EPA / Play",
    "off_success_rate": "Success Rate",
    "off_pass_rate": "Pass Rate",
    "off_turnover_rate": "Turnover Rate",
    "off_points_per_game": "Points / Game",
    "off_plays_per_game": "Plays / Game",
    "def_epa_allowed_per_play": "EPA Allowed / Play",
    "def_pass_epa_allowed_per_play": "Pass EPA Allowed / Play",
    "def_rush_epa_allowed_per_play": "Rush EPA Allowed / Play",
    "def_success_rate_allowed": "Success Rate Allowed",
    "def_sack_rate": "Sack Rate",
    "def_turnover_forced_rate": "Turnovers Forced / Play",
    "def_points_allowed_per_game": "Points Allowed / Game",
}

PERCENT_FEATURES = {
    "off_success_rate",
    "off_pass_rate",
    "off_turnover_rate",
    "def_success_rate_allowed",
    "def_sack_rate",
    "def_turnover_forced_rate",
}

st.set_page_config(
    page_title="SEC Team Rankings",
    page_icon="🏈",
    layout="wide",
)


# =============================================================================
# SportsDataverse normalization
# =============================================================================

def _as_pandas(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _series_or_default(
    df: pd.DataFrame,
    names: Iterable[str],
    default: object = np.nan,
) -> pd.Series:
    column = _first_existing(df, names)
    if column is None:
        return pd.Series(default, index=df.index)
    return df[column]


def _coerce_bool_series(
    series: pd.Series,
    default: bool = False,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(default, index=series.index, dtype=bool)

    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].ne(0)

    text = series.astype("string").str.strip().str.lower()
    result.loc[text.isin({"true", "t", "yes", "y", "1", "final", "completed"})] = True
    result.loc[text.isin({"false", "f", "no", "n", "0", "scheduled"})] = False
    return result


def normalize_schedule(raw_schedule) -> pd.DataFrame:
    raw = _as_pandas(raw_schedule)
    out = pd.DataFrame(index=raw.index)

    out["game_id"] = pd.to_numeric(
        _series_or_default(raw, ["game_id", "id"]),
        errors="coerce",
    )
    out["season"] = pd.to_numeric(
        _series_or_default(raw, ["season", "year"]),
        errors="coerce",
    )
    out["week"] = pd.to_numeric(
        _series_or_default(raw, ["week", "wk"]),
        errors="coerce",
    )
    out["start_date"] = pd.to_datetime(
        _series_or_default(raw, ["start_date", "game_date", "date"]),
        utc=True,
        errors="coerce",
    )

    out["home_id"] = pd.to_numeric(
        _series_or_default(raw, ["home_id", "home_team_id", "homeTeamId"]),
        errors="coerce",
    )
    out["away_id"] = pd.to_numeric(
        _series_or_default(raw, ["away_id", "away_team_id", "awayTeamId"]),
        errors="coerce",
    )

    out["home_team"] = _series_or_default(
        raw,
        ["home_team", "homeTeamName", "home_location"],
        default="Home",
    ).astype(str)
    out["away_team"] = _series_or_default(
        raw,
        ["away_team", "awayTeamName", "away_location"],
        default="Away",
    ).astype(str)

    out["home_abbreviation"] = _series_or_default(
        raw,
        ["home_abbreviation", "homeTeamAbbrev", "home_team_abbreviation"],
        default="",
    ).astype(str)
    out["away_abbreviation"] = _series_or_default(
        raw,
        ["away_abbreviation", "awayTeamAbbrev", "away_team_abbreviation"],
        default="",
    ).astype(str)

    out["home_score"] = pd.to_numeric(
        _series_or_default(raw, ["home_score", "home_points", "homeScore"]),
        errors="coerce",
    )
    out["away_score"] = pd.to_numeric(
        _series_or_default(raw, ["away_score", "away_points", "awayScore"]),
        errors="coerce",
    )

    completed_column = _first_existing(
        raw,
        ["completed", "status_type_completed", "status_completed"],
    )
    explicit_completed = pd.Series(False, index=raw.index, dtype=bool)
    if completed_column is not None:
        explicit_completed = _coerce_bool_series(raw[completed_column])

    score_completed = out["home_score"].notna() & out["away_score"].notna()
    out["completed"] = explicit_completed | score_completed

    out = out.dropna(
        subset=["game_id", "season", "week", "home_id", "away_id"]
    ).copy()

    out[["game_id", "season", "week", "home_id", "away_id"]] = out[
        ["game_id", "season", "week", "home_id", "away_id"]
    ].astype(int)

    return out.drop_duplicates("game_id").reset_index(drop=True)


def _fill_from_schedule(
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    target: str,
    schedule_column: str,
) -> None:
    mapping = schedule.set_index("game_id")[schedule_column]
    mapped = pbp["game_id"].map(mapping)

    if target not in pbp.columns:
        pbp[target] = mapped
    else:
        pbp[target] = pbp[target].where(pbp[target].notna(), mapped)


def _map_team_values_to_ids(
    pbp: pd.DataFrame,
    column: str,
) -> None:
    if column not in pbp.columns:
        return

    original = pbp[column]
    numeric = pd.to_numeric(original, errors="coerce")
    unresolved = numeric.isna() & original.notna()

    if not unresolved.any():
        pbp[column] = numeric
        return

    text = original.astype(str).str.strip().str.upper()

    home_candidates = []
    away_candidates = []

    for candidate in ["home", "homeTeamAbbrev", "homeTeamName", "home_team"]:
        if candidate in pbp.columns:
            home_candidates.append(
                pbp[candidate].astype(str).str.strip().str.upper()
            )

    for candidate in ["away", "awayTeamAbbrev", "awayTeamName", "away_team"]:
        if candidate in pbp.columns:
            away_candidates.append(
                pbp[candidate].astype(str).str.strip().str.upper()
            )

    for candidate in home_candidates:
        mask = unresolved & text.eq(candidate)
        numeric.loc[mask] = pd.to_numeric(
            pbp.loc[mask, "homeTeamId"],
            errors="coerce",
        )

    for candidate in away_candidates:
        mask = numeric.isna() & original.notna() & text.eq(candidate)
        numeric.loc[mask] = pd.to_numeric(
            pbp.loc[mask, "awayTeamId"],
            errors="coerce",
        )

    pbp[column] = numeric


def normalize_pbp_for_stats(
    raw_pbp,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adapt current/legacy SportsDataverse CFB PBP into the schema expected by
    stat_processing.py.
    """

    pbp = _as_pandas(raw_pbp)

    game_col = _first_existing(pbp, ["game_id", "id"])
    if game_col is None:
        raise KeyError("PBP is missing game_id")

    if game_col != "game_id":
        pbp["game_id"] = pbp[game_col]

    pbp["game_id"] = pd.to_numeric(pbp["game_id"], errors="coerce")
    pbp = pbp.dropna(subset=["game_id"]).copy()
    pbp["game_id"] = pbp["game_id"].astype(int)

    aliases = {
        "season": ["season", "year"],
        "week": ["week", "wk"],
        "homeTeamId": ["homeTeamId", "home_team_id", "home_id"],
        "awayTeamId": ["awayTeamId", "away_team_id", "away_id"],
        "homeTeamAbbrev": ["homeTeamAbbrev", "home_abbreviation"],
        "awayTeamAbbrev": ["awayTeamAbbrev", "away_abbreviation"],
        "homeScore": ["homeScore", "home_score", "home_points"],
        "awayScore": ["awayScore", "away_score", "away_points"],
        "EPA": ["EPA", "epa"],
        "sack": ["sack", "sack_vec"],
        "turnover_vec": ["turnover_vec", "turnover"],
        "status_type_completed": ["status_type_completed", "completed"],
    }

    for target, candidates in aliases.items():
        source = _first_existing(pbp, candidates)
        if target not in pbp.columns and source is not None:
            pbp[target] = pbp[source]

    _fill_from_schedule(pbp, schedule, "homeTeamId", "home_id")
    _fill_from_schedule(pbp, schedule, "awayTeamId", "away_id")
    _fill_from_schedule(pbp, schedule, "homeTeamAbbrev", "home_abbreviation")
    _fill_from_schedule(pbp, schedule, "awayTeamAbbrev", "away_abbreviation")
    _fill_from_schedule(pbp, schedule, "homeScore", "home_score")
    _fill_from_schedule(pbp, schedule, "awayScore", "away_score")
    _fill_from_schedule(pbp, schedule, "status_type_completed", "completed")

    if "season" not in pbp.columns:
        _fill_from_schedule(pbp, schedule, "season", "season")
    if "week" not in pbp.columns:
        _fill_from_schedule(pbp, schedule, "week", "week")

    # Prefer direct ESPN numeric possession ids when available.
    direct_pos_id = _first_existing(
        pbp,
        [
            "start.pos_team.id",
            "start_pos_team_id",
            "pos_team_id",
            "posteam_id",
        ],
    )
    direct_def_id = _first_existing(
        pbp,
        [
            "start.def_pos_team.id",
            "start_def_pos_team_id",
            "def_pos_team_id",
            "defteam_id",
        ],
    )

    if direct_pos_id is not None:
        pbp["pos_team"] = pd.to_numeric(
            pbp[direct_pos_id],
            errors="coerce",
        )
    elif "pos_team" not in pbp.columns:
        source = _first_existing(
            pbp,
            [
                "posteam",
                "possession_team",
                "offense_team",
                "start.pos_team.name",
            ],
        )
        if source is not None:
            pbp["pos_team"] = pbp[source]

    if direct_def_id is not None:
        pbp["def_pos_team"] = pd.to_numeric(
            pbp[direct_def_id],
            errors="coerce",
        )
    elif "def_pos_team" not in pbp.columns:
        source = _first_existing(
            pbp,
            [
                "defteam",
                "defense_team",
                "start.def_pos_team.name",
            ],
        )
        if source is not None:
            pbp["def_pos_team"] = pbp[source]

    _map_team_values_to_ids(pbp, "pos_team")
    _map_team_values_to_ids(pbp, "def_pos_team")

    possession_is_home_col = _first_existing(
        pbp,
        ["start.is_home", "is_home"],
    )
    if possession_is_home_col is not None:
        possession_is_home = _coerce_bool_series(
            pbp[possession_is_home_col]
        )

        pos_missing = pbp["pos_team"].isna()
        pbp.loc[pos_missing & possession_is_home, "pos_team"] = pbp.loc[
            pos_missing & possession_is_home,
            "homeTeamId",
        ]
        pbp.loc[pos_missing & ~possession_is_home, "pos_team"] = pbp.loc[
            pos_missing & ~possession_is_home,
            "awayTeamId",
        ]

        def_missing = pbp["def_pos_team"].isna()
        pbp.loc[def_missing & possession_is_home, "def_pos_team"] = pbp.loc[
            def_missing & possession_is_home,
            "awayTeamId",
        ]
        pbp.loc[def_missing & ~possession_is_home, "def_pos_team"] = pbp.loc[
            def_missing & ~possession_is_home,
            "homeTeamId",
        ]

    # Turnovers sometimes need to be reconstructed.
    if "turnover_vec" not in pbp.columns:
        interception = (
            _coerce_bool_series(pbp["int"])
            if "int" in pbp.columns
            else pd.Series(False, index=pbp.index)
        )
        fumble_lost = (
            _coerce_bool_series(pbp["fumble_lost"])
            if "fumble_lost" in pbp.columns
            else pd.Series(False, index=pbp.index)
        )
        pbp["turnover_vec"] = interception | fumble_lost

    required = [
        "game_id",
        "season",
        "week",
        "homeTeamId",
        "awayTeamId",
        "homeScore",
        "awayScore",
        "pos_team",
        "def_pos_team",
        "EPA",
        "pass",
        "rush",
        "sack",
        "turnover_vec",
    ]

    missing = [column for column in required if column not in pbp.columns]
    if missing:
        raise KeyError(
            "PBP could not be normalized for stat_processing.py. Missing: "
            + ", ".join(missing)
        )

    for column in [
        "season",
        "week",
        "homeTeamId",
        "awayTeamId",
        "homeScore",
        "awayScore",
        "pos_team",
        "def_pos_team",
        "EPA",
    ]:
        pbp[column] = pd.to_numeric(
            pbp[column],
            errors="coerce",
        )

    for column in ["pass", "rush", "sack", "turnover_vec"]:
        pbp[column] = _coerce_bool_series(pbp[column])

    if "penalty_no_play" in pbp.columns:
        pbp["penalty_no_play"] = _coerce_bool_series(
            pbp["penalty_no_play"]
        )

    score_completed = (
        pbp["homeScore"].notna()
        & pbp["awayScore"].notna()
    )

    if "status_type_completed" in pbp.columns:
        pbp["status_type_completed"] = (
            _coerce_bool_series(pbp["status_type_completed"])
            | score_completed
        )
    else:
        pbp["status_type_completed"] = score_completed

    return pbp


# =============================================================================
# Loading
# =============================================================================

def _load_cfb_schedule(seasons: list[int]) -> pd.DataFrame:
    try:
        raw = sdv.cfb.load_cfb_schedule(
            seasons=seasons,
            return_as_pandas=True,
        )
    except TypeError:
        raw = sdv.cfb.load_cfb_schedule(seasons=seasons)
    return _as_pandas(raw)


def _load_cfb_pbp(seasons: list[int]) -> pd.DataFrame:
    loader = getattr(sdv.cfb, "load_cfb_pbp", None)

    if loader is None:
        loader = getattr(sdv.cfb, "load_cfb_pbp_r", None)

    if loader is None:
        raise AttributeError(
            "sportsdataverse.cfb exposes neither load_cfb_pbp nor load_cfb_pbp_r"
        )

    try:
        raw = loader(
            seasons=seasons,
            return_as_pandas=True,
        )
    except TypeError:
        raw = loader(seasons=seasons)

    return _as_pandas(raw)


@st.cache_data(ttl=3600, show_spinner=False)
def load_rankings_data(
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Previous season is included so Week 1 / early season rankings have history.
    seasons = [season - 1, season]

    schedule = normalize_schedule(
        _load_cfb_schedule(seasons)
    )

    pbp = normalize_pbp_for_stats(
        _load_cfb_pbp(seasons),
        schedule,
    )

    return pbp, schedule


# =============================================================================
# Feature calculation
# =============================================================================

def get_default_week(
    schedule: pd.DataFrame,
    season: int,
) -> int:
    games = schedule[
        schedule["season"] == season
    ].dropna(subset=["week"]).copy()

    if games.empty:
        raise ValueError(
            f"No schedule data is available for {season}."
        )

    now = pd.Timestamp.now(tz="UTC")

    if games["start_date"].notna().any():
        week_dates = games.groupby("week")["start_date"].agg(
            ["min", "max"]
        )

        for week, row in week_dates.iterrows():
            if pd.notna(row["min"]) and pd.notna(row["max"]):
                if (
                    row["min"] - pd.Timedelta(days=2)
                    <= now
                    <= row["max"] + pd.Timedelta(days=2)
                ):
                    return int(week)

        midpoint = (
            week_dates["min"]
            + (week_dates["max"] - week_dates["min"]) / 2
        )
        return int((midpoint - now).abs().idxmin())

    return int(games["week"].max())


def _team_game_ids(
    pbp: pd.DataFrame,
    team_id: int,
) -> list[int]:
    games = (
        pbp[
            ["game_id", "homeTeamId", "awayTeamId"]
        ]
        .dropna()
        .drop_duplicates("game_id")
    )

    mask = (
        pd.to_numeric(games["homeTeamId"], errors="coerce").eq(team_id)
        | pd.to_numeric(games["awayTeamId"], errors="coerce").eq(team_id)
    )

    return (
        pd.to_numeric(
            games.loc[mask, "game_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )


def build_sec_feature_table(
    pbp: pd.DataFrame,
    season: int,
    week: int,
    lookback_games: int,
    progress_callback=None,
) -> pd.DataFrame:
    rows = []
    total = len(SEC_TEAMS)

    for index, (team_id, (team_name, abbreviation)) in enumerate(
        SEC_TEAMS.items(),
        start=1,
    ):
        game_ids = _team_game_ids(pbp, team_id)
        team_pbp = pbp[
            pbp["game_id"].isin(game_ids)
        ].copy()

        offense = stat_processing.get_offensive_team_stats_past_n_games(
            team=team_id,
            n=lookback_games,
            pbp_data=team_pbp,
            before_season=season,
            before_week=week,
        )

        defense = stat_processing.get_defensive_team_stats_past_n_games(
            team=team_id,
            n=lookback_games,
            pbp_data=team_pbp,
            before_season=season,
            before_week=week,
        )

        rows.append(
            {
                "team_id": team_id,
                "Team": team_name,
                "Abbreviation": abbreviation,
                **offense,
                **defense,
            }
        )

        if progress_callback is not None:
            progress_callback(
                int(100 * index / total),
                f"Calculating {team_name}: {index}/{total} SEC teams",
            )

    return pd.DataFrame(rows)


# =============================================================================
# Ranking
# =============================================================================

def _quality_percentile(
    series: pd.Series,
    direction: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if direction == "higher":
        return numeric.rank(
            pct=True,
            ascending=True,
            method="average",
        ) * 100

    if direction == "lower":
        return numeric.rank(
            pct=True,
            ascending=False,
            method="average",
        ) * 100

    raise ValueError(
        f"Unknown ranking direction: {direction}"
    )


def _feature_rank(
    series: pd.Series,
    direction: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    return numeric.rank(
        ascending=(direction == "lower"),
        method="min",
    ).astype("Int64")


def add_rankings(
    features: pd.DataFrame,
) -> pd.DataFrame:
    ranked = features.copy()

    offense_percentiles = []
    defense_percentiles = []

    for feature, direction in OFFENSE_QUALITY.items():
        percentile_column = f"{feature}__percentile"
        rank_column = f"{feature}__rank"

        ranked[percentile_column] = _quality_percentile(
            ranked[feature],
            direction,
        )
        ranked[rank_column] = _feature_rank(
            ranked[feature],
            direction,
        )

        offense_percentiles.append(percentile_column)

    for feature, direction in DEFENSE_QUALITY.items():
        percentile_column = f"{feature}__percentile"
        rank_column = f"{feature}__rank"

        ranked[percentile_column] = _quality_percentile(
            ranked[feature],
            direction,
        )
        ranked[rank_column] = _feature_rank(
            ranked[feature],
            direction,
        )

        defense_percentiles.append(percentile_column)

    ranked["Offense Score"] = ranked[
        offense_percentiles
    ].mean(axis=1)

    ranked["Defense Score"] = ranked[
        defense_percentiles
    ].mean(axis=1)

    ranked["Overall Score"] = (
        ranked["Offense Score"]
        + ranked["Defense Score"]
    ) / 2

    ranked["Offense Rank"] = ranked[
        "Offense Score"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    ranked["Defense Rank"] = ranked[
        "Defense Score"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    ranked["Overall Rank"] = ranked[
        "Overall Score"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    # Descriptive ranks. These are shown but do not affect composite scores.
    ranked["Pass Rate Rank"] = ranked[
        "off_pass_rate"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    ranked["Pace Rank"] = ranked[
        "off_plays_per_game"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    return ranked


# =============================================================================
# Display helpers
# =============================================================================

def _format_feature_value(
    feature: str,
    value: float,
) -> str:
    if pd.isna(value):
        return "—"

    if feature in PERCENT_FEATURES:
        return f"{100 * value:.1f}%"

    if feature in {
        "off_points_per_game",
        "off_plays_per_game",
        "def_points_allowed_per_game",
    }:
        return f"{value:.1f}"

    return f"{value:.3f}"


def offense_display_table(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    table = rankings.sort_values(
        "Offense Rank"
    ).copy()

    columns = {
        "Offense Rank": "Rank",
        "Team": "Team",
        "Offense Score": "Score",
        "off_epa_per_play": "EPA/Play",
        "off_pass_epa_per_play": "Pass EPA",
        "off_rush_epa_per_play": "Rush EPA",
        "off_success_rate": "Success %",
        "off_turnover_rate": "Turnover %",
        "off_points_per_game": "PPG",
        "off_pass_rate": "Pass %",
        "off_plays_per_game": "Plays/Game",
    }

    output = table[list(columns)].rename(
        columns=columns
    )

    output["Score"] = output["Score"].round(1)

    for column in ["EPA/Play", "Pass EPA", "Rush EPA"]:
        output[column] = output[column].round(3)

    for column in ["Success %", "Turnover %", "Pass %"]:
        output[column] = (
            100 * output[column]
        ).round(1)

    for column in ["PPG", "Plays/Game"]:
        output[column] = output[column].round(1)

    return output


def defense_display_table(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    table = rankings.sort_values(
        "Defense Rank"
    ).copy()

    columns = {
        "Defense Rank": "Rank",
        "Team": "Team",
        "Defense Score": "Score",
        "def_epa_allowed_per_play": "EPA Allowed",
        "def_pass_epa_allowed_per_play": "Pass EPA Allowed",
        "def_rush_epa_allowed_per_play": "Rush EPA Allowed",
        "def_success_rate_allowed": "Success Allowed %",
        "def_sack_rate": "Sack %",
        "def_turnover_forced_rate": "Takeaway %",
        "def_points_allowed_per_game": "Points Allowed",
    }

    output = table[list(columns)].rename(
        columns=columns
    )

    output["Score"] = output["Score"].round(1)

    for column in [
        "EPA Allowed",
        "Pass EPA Allowed",
        "Rush EPA Allowed",
    ]:
        output[column] = output[column].round(3)

    for column in [
        "Success Allowed %",
        "Sack %",
        "Takeaway %",
    ]:
        output[column] = (
            100 * output[column]
        ).round(1)

    output["Points Allowed"] = output[
        "Points Allowed"
    ].round(1)

    return output


def feature_rank_detail(
    row: pd.Series,
    feature_map: dict[str, str],
) -> pd.DataFrame:
    records = []

    for feature, direction in feature_map.items():
        records.append(
            {
                "Metric": LABELS[feature],
                "Value": _format_feature_value(
                    feature,
                    row[feature],
                ),
                "SEC Rank": int(
                    row[f"{feature}__rank"]
                ),
                "Better": (
                    "Higher"
                    if direction == "higher"
                    else "Lower"
                ),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# Streamlit UI
# =============================================================================

st.title("🏈 SEC Team Rankings")
st.caption(
    "Ranks all 16 SEC teams using recent offensive and defensive "
    "SportsDataverse play-by-play features."
)

with st.sidebar:
    st.header("Ranking settings")

    season = st.number_input(
        "Season",
        min_value=2020,
        max_value=DEFAULT_SEASON + 1,
        value=DEFAULT_SEASON,
        step=1,
    )

with st.spinner("Loading SportsDataverse schedule and play-by-play..."):
    pbp, schedule = load_rankings_data(
        int(season)
    )

available_weeks = sorted(
    schedule[
        schedule["season"] == int(season)
    ]["week"]
    .dropna()
    .astype(int)
    .unique()
)

if not available_weeks:
    st.error(
        f"No schedule weeks were found for {int(season)}."
    )
    st.stop()

try:
    default_week = get_default_week(
        schedule,
        int(season),
    )
except ValueError:
    default_week = available_weeks[0]

with st.sidebar:
    week = st.selectbox(
        "Rank teams entering Week",
        available_weeks,
        index=(
            available_weeks.index(default_week)
            if default_week in available_weeks
            else 0
        ),
        help=(
            "Week N uses only games before Week N, so the rankings are "
            "leakage-safe."
        ),
    )

    lookback_games = st.slider(
        "Games in recent-form window",
        min_value=3,
        max_value=15,
        value=DEFAULT_LOOKBACK_GAMES,
        help=(
            "The most recent N completed games are used. "
            "Previous-season games are available early in the season."
        ),
    )

st.info(
    f"These rankings represent information available **before "
    f"{int(season)} Week {int(week)}**, using each team's most recent "
    f"**{int(lookback_games)} completed games**."
)

ranking_progress = st.progress(
    0,
    text="Calculating SEC team features...",
)

def update_ranking_progress(
    value: int,
    message: str,
) -> None:
    ranking_progress.progress(
        min(max(value, 0), 100),
        text=message,
    )

feature_table = build_sec_feature_table(
    pbp=pbp,
    season=int(season),
    week=int(week),
    lookback_games=int(lookback_games),
    progress_callback=update_ranking_progress,
)

rankings = add_rankings(
    feature_table
)

ranking_progress.progress(
    100,
    text="SEC rankings complete.",
)

# Warn if teams have no usable historical signal.
quality_columns = [
    *OFFENSE_QUALITY.keys(),
    *DEFENSE_QUALITY.keys(),
]

no_signal = rankings[
    rankings[quality_columns].abs().sum(axis=1) == 0
]["Team"].tolist()

if no_signal:
    st.warning(
        "No usable historical feature signal was found for: "
        + ", ".join(no_signal)
        + ". Those teams may not have enough PBP before the selected week."
    )

overall_tab, offense_tab, defense_tab, detail_tab, method_tab = st.tabs(
    [
        "Overall",
        "Offense",
        "Defense",
        "Team Detail",
        "Methodology",
    ]
)

with overall_tab:
    st.subheader("SEC Overall Rankings")

    overall = (
        rankings[
            [
                "Overall Rank",
                "Team",
                "Overall Score",
                "Offense Rank",
                "Offense Score",
                "Defense Rank",
                "Defense Score",
            ]
        ]
        .sort_values("Overall Rank")
        .copy()
    )

    for column in [
        "Overall Score",
        "Offense Score",
        "Defense Score",
    ]:
        overall[column] = overall[column].round(1)

    st.dataframe(
        overall,
        hide_index=True,
        use_container_width=True,
    )

    chart = (
        overall[
            [
                "Team",
                "Overall Score",
            ]
        ]
        .set_index("Team")
    )

    st.bar_chart(chart)

with offense_tab:
    st.subheader("SEC Offensive Rankings")
    st.caption(
        "Composite uses EPA/play, pass EPA, rush EPA, success rate, "
        "turnover rate, and points/game. Pass rate and plays/game are "
        "shown as style/pace indicators but are not included in the score."
    )

    st.dataframe(
        offense_display_table(rankings),
        hide_index=True,
        use_container_width=True,
    )

    offense_chart = (
        rankings[
            ["Team", "Offense Score"]
        ]
        .sort_values("Offense Score")
        .set_index("Team")
    )
    st.bar_chart(offense_chart)

with defense_tab:
    st.subheader("SEC Defensive Rankings")
    st.caption(
        "Lower EPA allowed, success allowed, and points allowed are better; "
        "higher sack and turnover-forced rates are better."
    )

    st.dataframe(
        defense_display_table(rankings),
        hide_index=True,
        use_container_width=True,
    )

    defense_chart = (
        rankings[
            ["Team", "Defense Score"]
        ]
        .sort_values("Defense Score")
        .set_index("Team")
    )
    st.bar_chart(defense_chart)

with detail_tab:
    selected_team = st.selectbox(
        "Team",
        rankings.sort_values("Overall Rank")["Team"].tolist(),
    )

    team = rankings[
        rankings["Team"] == selected_team
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Overall SEC Rank",
        f'#{int(team["Overall Rank"])}',
        f'{team["Overall Score"]:.1f} / 100',
    )
    c2.metric(
        "Offense SEC Rank",
        f'#{int(team["Offense Rank"])}',
        f'{team["Offense Score"]:.1f} / 100',
    )
    c3.metric(
        "Defense SEC Rank",
        f'#{int(team["Defense Rank"])}',
        f'{team["Defense Score"]:.1f} / 100',
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Offensive feature ranks")
        st.dataframe(
            feature_rank_detail(
                team,
                OFFENSE_QUALITY,
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### Offensive style")
        style = pd.DataFrame(
            [
                {
                    "Metric": "Pass Rate",
                    "Value": _format_feature_value(
                        "off_pass_rate",
                        team["off_pass_rate"],
                    ),
                    "SEC Rank": int(
                        team["Pass Rate Rank"]
                    ),
                },
                {
                    "Metric": "Plays / Game",
                    "Value": _format_feature_value(
                        "off_plays_per_game",
                        team["off_plays_per_game"],
                    ),
                    "SEC Rank": int(
                        team["Pace Rank"]
                    ),
                },
            ]
        )
        st.dataframe(
            style,
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.markdown("#### Defensive feature ranks")
        st.dataframe(
            feature_rank_detail(
                team,
                DEFENSE_QUALITY,
            ),
            hide_index=True,
            use_container_width=True,
        )

with method_tab:
    st.subheader("How the rankings work")

    st.markdown(
        """
Each quality metric is ranked **only against the other 15 SEC teams**.

For a metric where higher is better, such as offensive EPA/play, the best
team receives the highest percentile score. For a metric where lower is
better, such as defensive EPA allowed/play, the direction is reversed.

**Offense Score**
- EPA/play
- Pass EPA/play
- Rush EPA/play
- Success rate
- Turnover rate — lower is better
- Points/game

**Not included in the offense composite**
- Pass rate — offensive style
- Plays/game — pace

**Defense Score**
- EPA allowed/play — lower is better
- Pass EPA allowed/play — lower is better
- Rush EPA allowed/play — lower is better
- Success rate allowed — lower is better
- Sack rate — higher is better
- Turnovers forced/play — higher is better
- Points allowed/game — lower is better

The offense and defense scores are the equal-weight average of their
feature percentile scores. The **Overall Score** is the equal-weight
average of Offense Score and Defense Score.

A 100 score therefore means a team is near the top of the SEC across the
included metrics; it is a relative conference score, not an absolute
football rating.
        """
    )
