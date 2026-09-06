"""CFB feature engineering for SportsDataverse play-by-play data.

This module is a replacement for the NFL-specific ``stat-processing.py``.
It is designed around the columns returned by ``sportsdataverse.cfb.load_cfb_pbp``.

The important rule is that every prediction feature can be calculated using only
GAMES BEFORE the season/week being predicted. This prevents data leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature order used when a NumPy/list model input is needed.
# ---------------------------------------------------------------------------

OFFENSIVE_FEATURES = [
    "off_epa_per_play",
    "off_pass_epa_per_play",
    "off_rush_epa_per_play",
    "off_success_rate",
    "off_pass_rate",
    "off_turnover_rate",
    "off_points_per_game",
    "off_plays_per_game",
]

DEFENSIVE_FEATURES = [
    "def_epa_allowed_per_play",
    "def_pass_epa_allowed_per_play",
    "def_rush_epa_allowed_per_play",
    "def_success_rate_allowed",
    "def_sack_rate",
    "def_turnover_forced_rate",
    "def_points_allowed_per_game",
]

MATCHUP_FEATURES = ["is_home", *OFFENSIVE_FEATURES, *DEFENSIVE_FEATURES]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_pandas(data: Any) -> pd.DataFrame:
    """Return a defensive copy as a pandas DataFrame.

    SportsDataverse commonly returns Polars DataFrames. Keeping this conversion
    here lets the rest of this module use one implementation while accepting
    either pandas or Polars input.
    """

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    raise TypeError("pbp_data must be a pandas or Polars DataFrame")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "SportsDataverse play-by-play data is missing required columns: "
            + ", ".join(missing)
        )


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def _safe_rate(series: pd.Series, default: float = 0.0) -> float:
    """Mean of a boolean/0-1 series."""

    if series.empty:
        return default

    values = series.fillna(False).astype(bool)
    return float(values.mean())


def _team_lookup_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """Build an abbreviation -> ESPN team id table from PBP metadata."""

    _require_columns(
        pbp,
        ["homeTeamId", "awayTeamId", "homeTeamAbbrev", "awayTeamAbbrev"],
    )

    home = pbp[["homeTeamId", "homeTeamAbbrev"]].rename(
        columns={"homeTeamId": "team_id", "homeTeamAbbrev": "abbreviation"}
    )
    away = pbp[["awayTeamId", "awayTeamAbbrev"]].rename(
        columns={"awayTeamId": "team_id", "awayTeamAbbrev": "abbreviation"}
    )

    teams = pd.concat([home, away], ignore_index=True).dropna()
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce")
    teams = teams.dropna(subset=["team_id"])
    teams["team_id"] = teams["team_id"].astype(int)
    teams["abbreviation"] = teams["abbreviation"].astype(str).str.upper()

    return teams.drop_duplicates(["team_id", "abbreviation"])


def resolve_team_id(team: int | str, pbp_data: Any) -> int:
    """Resolve either an ESPN team id or an abbreviation to an ESPN team id.

    Examples
    --------
    ``resolve_team_id(333, pbp)``
    ``resolve_team_id("ALA", pbp)``
    """

    pbp = _to_pandas(pbp_data)

    if isinstance(team, (int, np.integer)):
        return int(team)

    text = str(team).strip()
    if text.isdigit():
        return int(text)

    lookup = _team_lookup_table(pbp)
    matches = lookup[lookup["abbreviation"] == text.upper()]["team_id"].unique()

    if len(matches) == 0:
        raise ValueError(f"Could not find team abbreviation {team!r} in PBP data")
    if len(matches) > 1:
        raise ValueError(
            f"Team abbreviation {team!r} maps to multiple ESPN team ids: "
            f"{matches.tolist()}. Pass the numeric team id instead."
        )

    return int(matches[0])


def _build_game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """Collapse play-by-play into one row per game."""
    pbp = pbp.reset_index(drop=True)

    required = [
        "game_id",
        "season",
        "week",
        "homeTeamId",
        "awayTeamId",
        "homeScore",
        "awayScore",
    ]
    _require_columns(pbp, required)

    aggregations: dict[str, str] = {
        "season": "first",
        "week": "first",
        "homeTeamId": "first",
        "awayTeamId": "first",
        "homeScore": "max",
        "awayScore": "max",
    }

    if "homeTeamAbbrev" in pbp.columns:
        aggregations["homeTeamAbbrev"] = "first"
    if "awayTeamAbbrev" in pbp.columns:
        aggregations["awayTeamAbbrev"] = "first"
    if "status_type_completed" in pbp.columns:
        aggregations["status_type_completed"] = "max"

    games = pbp.groupby("game_id", as_index=False).agg(aggregations)

    for column in ["game_id", "season", "week", "homeTeamId", "awayTeamId"]:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    return games.dropna(subset=["game_id", "season", "week"])


def _historical_games(
    team: int | str,
    n: int,
    pbp_data: Any,
    before_season: int | None = None,
    before_week: int | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Return team id, all PBP, and the team's most recent N completed games."""

    if n <= 0:
        raise ValueError("n must be greater than 0")

    pbp = _to_pandas(pbp_data)
    team_id = resolve_team_id(team, pbp)
    games = _build_game_table(pbp)

    team_games = games[
        (games["homeTeamId"] == team_id) | (games["awayTeamId"] == team_id)
    ].copy()

    # Prefer the explicit completion flag, but final scores are also sufficient
    # evidence that a historical game was played. Some SportsDataverse release
    # variants have incomplete/stale status metadata while still carrying scores.
    score_completed = team_games["homeScore"].notna() & team_games["awayScore"].notna()

    if "status_type_completed" in team_games.columns:
        status = team_games["status_type_completed"]
        if not pd.api.types.is_bool_dtype(status):
            text = status.astype("string").str.strip().str.lower()
            numeric = pd.to_numeric(status, errors="coerce")
            status_bool = numeric.fillna(0).ne(0)
            status_bool = status_bool | text.isin(
                {"true", "t", "yes", "y", "1", "final", "status_final", "completed"}
            )
        else:
            status_bool = status.fillna(False).astype(bool)

        team_games = team_games[status_bool | score_completed]
    else:
        team_games = team_games[score_completed]

    # Only use games strictly before the week being predicted.
    if before_season is not None:
        if before_week is None:
            team_games = team_games[team_games["season"] < before_season]
        else:
            team_games = team_games[
                (team_games["season"] < before_season)
                | (
                    (team_games["season"] == before_season)
                    & (team_games["week"] < before_week)
                )
            ]

    team_games = team_games.sort_values(
        ["season", "week", "game_id"], ascending=False
    ).head(n)

    return team_id, pbp, team_games


def _scrimmage_plays(pbp: pd.DataFrame, game_ids: list[int]) -> pd.DataFrame:
    """Return pass/rush plays from the requested games, removing no-plays."""

    _require_columns(pbp, ["game_id", "pass", "rush", "EPA"])

    plays = pbp[pbp["game_id"].isin(game_ids)].copy()

    if "penalty_no_play" in plays.columns:
        plays = plays[~plays["penalty_no_play"].fillna(False)]

    is_pass = plays["pass"].fillna(False).astype(bool)
    is_rush = plays["rush"].fillna(False).astype(bool)

    return plays[is_pass | is_rush].copy()


def _points_for_team(team_id: int, games: pd.DataFrame) -> tuple[float, float]:
    """Return (points scored/game, points allowed/game)."""

    if games.empty:
        return 0.0, 0.0

    home_mask = games["homeTeamId"] == team_id

    points_scored = np.where(home_mask, games["homeScore"], games["awayScore"])
    points_allowed = np.where(home_mask, games["awayScore"], games["homeScore"])

    return (
        float(pd.to_numeric(pd.Series(points_scored), errors="coerce").mean()),
        float(pd.to_numeric(pd.Series(points_allowed), errors="coerce").mean()),
    )


# ---------------------------------------------------------------------------
# Offensive team stats
# ---------------------------------------------------------------------------


def get_offensive_team_stats_past_n_games(
    team: int | str,
    n: int,
    pbp_data: Any,
    before_season: int | None = None,
    before_week: int | None = None,
) -> dict[str, float]:
    """Calculate recent offensive CFB features for one team.

    Parameters
    ----------
    team:
        ESPN team id or ESPN abbreviation.
    n:
        Number of most recent completed games to use.
    pbp_data:
        Data returned by ``sportsdataverse.cfb.load_cfb_pbp``.
    before_season, before_week:
        Optional prediction cutoff. For example, ``2026, 6`` means use only
        games before Week 6 of the 2026 season.
    """

    team_id, pbp, games = _historical_games(
        team, n, pbp_data, before_season, before_week
    )

    if games.empty:
        return {feature: 0.0 for feature in OFFENSIVE_FEATURES}

    _require_columns(pbp, ["pos_team", "turnover_vec"])

    game_ids = games["game_id"].astype(int).tolist()
    plays = _scrimmage_plays(pbp, game_ids)
    offense = plays[pd.to_numeric(plays["pos_team"], errors="coerce") == team_id]

    if offense.empty:
        return {feature: 0.0 for feature in OFFENSIVE_FEATURES}

    pass_plays = offense[offense["pass"].fillna(False).astype(bool)]
    rush_plays = offense[offense["rush"].fillna(False).astype(bool)]

    points_scored, _ = _points_for_team(team_id, games)

    stats = {
        "off_epa_per_play": _safe_mean(offense["EPA"]),
        "off_pass_epa_per_play": _safe_mean(pass_plays["EPA"]),
        "off_rush_epa_per_play": _safe_mean(rush_plays["EPA"]),
        "off_success_rate": float(
            (pd.to_numeric(offense["EPA"], errors="coerce") > 0).mean()
        ),
        "off_pass_rate": float(offense["pass"].fillna(False).astype(bool).mean()),
        "off_turnover_rate": _safe_rate(offense["turnover_vec"]),
        "off_points_per_game": points_scored,
        "off_plays_per_game": float(len(offense) / len(games)),
    }

    return stats


# ---------------------------------------------------------------------------
# Defensive team stats
# ---------------------------------------------------------------------------


def get_defensive_team_stats_past_n_games(
    team: int | str,
    n: int,
    pbp_data: Any,
    before_season: int | None = None,
    before_week: int | None = None,
) -> dict[str, float]:
    """Calculate recent defensive CFB features for one team.

    This replaces the old NFL implementation that depended on ``defteam``,
    ``special_teams_play``, ``passing_yards`` and ``rushing_yards``.
    """

    team_id, pbp, games = _historical_games(
        team, n, pbp_data, before_season, before_week
    )

    if games.empty:
        return {feature: 0.0 for feature in DEFENSIVE_FEATURES}

    _require_columns(pbp, ["def_pos_team", "sack", "turnover_vec"])

    game_ids = games["game_id"].astype(int).tolist()
    plays = _scrimmage_plays(pbp, game_ids)
    defense = plays[pd.to_numeric(plays["def_pos_team"], errors="coerce") == team_id]

    if defense.empty:
        return {feature: 0.0 for feature in DEFENSIVE_FEATURES}

    pass_plays = defense[defense["pass"].fillna(False).astype(bool)]
    rush_plays = defense[defense["rush"].fillna(False).astype(bool)]

    _, points_allowed = _points_for_team(team_id, games)

    sack_rate = 0.0
    if len(pass_plays) > 0:
        sack_rate = float(pass_plays["sack"].fillna(False).astype(bool).mean())

    stats = {
        # EPA is from the offense's perspective, so lower is better for defense.
        "def_epa_allowed_per_play": _safe_mean(defense["EPA"]),
        "def_pass_epa_allowed_per_play": _safe_mean(pass_plays["EPA"]),
        "def_rush_epa_allowed_per_play": _safe_mean(rush_plays["EPA"]),
        "def_success_rate_allowed": float(
            (pd.to_numeric(defense["EPA"], errors="coerce") > 0).mean()
        ),
        "def_sack_rate": sack_rate,
        "def_turnover_forced_rate": _safe_rate(defense["turnover_vec"]),
        "def_points_allowed_per_game": points_allowed,
    }

    return stats


# ---------------------------------------------------------------------------
# Matchup/model helpers
# ---------------------------------------------------------------------------


def get_matchup_features(
    offense_team: int | str,
    defense_team: int | str,
    is_home: bool,
    n: int,
    pbp_data: Any,
    before_season: int | None = None,
    before_week: int | None = None,
) -> dict[str, float]:
    """Build one score-prediction feature row for a team in a matchup."""

    offense = get_offensive_team_stats_past_n_games(
        offense_team,
        n,
        pbp_data,
        before_season=before_season,
        before_week=before_week,
    )

    opponent_defense = get_defensive_team_stats_past_n_games(
        defense_team,
        n,
        pbp_data,
        before_season=before_season,
        before_week=before_week,
    )

    return {
        "is_home": float(bool(is_home)),
        **offense,
        **opponent_defense,
    }


def get_matchup_feature_vector(
    offense_team: int | str,
    defense_team: int | str,
    is_home: bool,
    n: int,
    pbp_data: Any,
    before_season: int | None = None,
    before_week: int | None = None,
) -> list[float]:
    """Return matchup features in a stable order for sklearn/Keras models."""

    features = get_matchup_features(
        offense_team,
        defense_team,
        is_home,
        n,
        pbp_data,
        before_season=before_season,
        before_week=before_week,
    )

    return [float(features[name]) for name in MATCHUP_FEATURES]


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------


def get_all_current_teams(
    pbp_data: Any,
    season: int | None = None,
) -> list[str]:
    """Return all team abbreviations represented in the selected CFB season."""

    pbp = _to_pandas(pbp_data)

    if season is None:
        season = int(pd.to_numeric(pbp["season"], errors="coerce").max())

    season_data = pbp[pd.to_numeric(pbp["season"], errors="coerce") == season]

    teams = pd.concat(
        [season_data["homeTeamAbbrev"], season_data["awayTeamAbbrev"]],
        ignore_index=True,
    )

    return sorted(teams.dropna().astype(str).unique().tolist())
