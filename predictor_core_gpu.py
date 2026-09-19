"""Reusable StockPig prediction/model functions.

Generated from predictor_model.py so other Streamlit pages can reuse the same
SportsDataverse normalization, leakage-safe feature engineering, Random Forest
training, validation residuals, and Monte Carlo simulation without executing
the predictor page UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sportsdataverse as sdv
import stat_processing

try:
    import torch
except ImportError:  # CPU-only fallback if PyTorch is not installed.
    torch = None


def _detect_torch_device():
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
    except Exception:
        pass
    return torch.device("cpu")


TORCH_DEVICE = _detect_torch_device()
GPU_ACCELERATION_AVAILABLE = bool(
    torch is not None
    and TORCH_DEVICE is not None
    and TORCH_DEVICE.type == "cuda"
)


def accelerator_name() -> str:
    if GPU_ACCELERATION_AVAILABLE:
        try:
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return "ROCm GPU"
    return "CPU"



# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SEASON = datetime.now(timezone.utc).year
TRAIN_START_SEASON = 2024
LOOKBACK_GAMES = 8
VALIDATION_FRACTION = 0.20
MONTE_CARLO_SIMS = int(1e7)
RANDOM_SEED = 42

if torch.cuda.is_available():
    MONTE_CARLO_SIMS = 1_000_000
else:
    MONTE_CARLO_SIMS = 30_000
# Current SEC membership. Numeric ids are ESPN team ids.
# These are used as a fallback when a schedule loader does not expose conference
# names directly.
SEC_TEAM_IDS = {
    333,   # Alabama
    8,     # Arkansas
    2,     # Auburn
    57,    # Florida
    61,    # Georgia
    96,    # Kentucky
    99,    # LSU
    344,   # Mississippi State
    142,   # Missouri
    201,   # Oklahoma
    145,   # Ole Miss
    2579,  # South Carolina
    2633,  # Tennessee
    251,   # Texas
    245,   # Texas A&M
    238,   # Vanderbilt
}

FEATURE_COLUMNS = list(stat_processing.MATCHUP_FEATURES)



# =============================================================================
# Data normalization
# =============================================================================


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


# def _series_or_default(
    # df: pd.DataFrame,
    # names: Iterable[str],
    # default=np.nan,
# ) -> pd.Series:
    # column = _first_existing(df, names)
    # if column is None:
        # return pd.Series(default, index=df.index)
    # return df[column]
    
def _series_or_default(
    df: pd.DataFrame,
    names: Iterable[str],
    default: object = np.nan,
) -> pd.Series:
    column = _first_existing(df, names)

    if column is None:
        return pd.Series(default, index=df.index)

    return df[column]

def _as_pandas(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _coerce_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    """Coerce common bool/int/string representations without treating "False" as True."""
    if series is None:
        return pd.Series(dtype=bool)

    values = series.copy()

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)

    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(default, index=values.index, dtype=bool)

    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].ne(0)

    text = values.astype("string").str.strip().str.lower()
    true_values = {"true", "t", "yes", "y", "1", "final", "status_final", "completed"}
    false_values = {"false", "f", "no", "n", "0", "scheduled", "status_scheduled"}

    result.loc[text.isin(true_values)] = True
    result.loc[text.isin(false_values)] = False

    return result


def normalize_schedule(raw_schedule) -> pd.DataFrame:
    """Normalize SportsDataverse schedule variants into one schema."""

    raw = _as_pandas(raw_schedule)
    out = pd.DataFrame(index=raw.index)

    out["game_id"] = pd.to_numeric(
        _series_or_default(raw, ["game_id", "id"]), errors="coerce"
    )
    out["season"] = pd.to_numeric(
        _series_or_default(raw, ["season", "year"]), errors="coerce"
    )
    out["week"] = pd.to_numeric(
        _series_or_default(raw, ["week", "wk"]), errors="coerce"
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

    # Treat schedule completion as redundant evidence rather than a single
    # point of failure. SportsDataverse exposes both a completion flag and final
    # score columns, and historical releases can differ slightly in which field
    # is populated.
    completed_col = _first_existing(
        raw,
        ["completed", "status_type_completed", "status_completed"],
    )

    explicit_completed = pd.Series(False, index=raw.index, dtype=bool)
    if completed_col is not None:
        explicit_completed = _coerce_bool_series(raw[completed_col])

    score_completed = out["home_score"].notna() & out["away_score"].notna()

    status_col = _first_existing(
        raw,
        ["status", "status_type_name", "status_name"],
    )
    status_completed = pd.Series(False, index=raw.index, dtype=bool)
    if status_col is not None:
        status_text = raw[status_col].astype("string").str.strip().str.upper()
        status_completed = status_text.isin(
            {"FINAL", "STATUS_FINAL", "COMPLETED", "STATUS_COMPLETED"}
        )

    out["completed"] = explicit_completed | score_completed | status_completed

    out["home_conference"] = _series_or_default(
        raw,
        ["home_conference", "home_team_conference"],
        default="",
    ).astype(str)
    out["away_conference"] = _series_or_default(
        raw,
        ["away_conference", "away_team_conference"],
        default="",
    ).astype(str)

    out["neutral_site"] = _series_or_default(
        raw,
        ["neutral_site", "neutralSite"],
        default=False,
    ).fillna(False).astype(bool)

    out = out.dropna(subset=["game_id", "season", "week", "home_id", "away_id"])
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
    """Coerce possession-team fields to ESPN numeric ids when necessary."""

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
            home_candidates.append(pbp[candidate].astype(str).str.strip().str.upper())

    for candidate in ["away", "awayTeamAbbrev", "awayTeamName", "away_team"]:
        if candidate in pbp.columns:
            away_candidates.append(pbp[candidate].astype(str).str.strip().str.upper())

    for candidate in home_candidates:
        mask = unresolved & text.eq(candidate)
        numeric.loc[mask] = pd.to_numeric(pbp.loc[mask, "homeTeamId"], errors="coerce")

    for candidate in away_candidates:
        mask = numeric.isna() & original.notna() & text.eq(candidate)
        numeric.loc[mask] = pd.to_numeric(pbp.loc[mask, "awayTeamId"], errors="coerce")

    pbp[column] = numeric


def normalize_pbp_for_stats(raw_pbp, schedule: pd.DataFrame) -> pd.DataFrame:
    """Make PBP compatible with the current stat_processing.py contract.

    The SportsDataverse CFB datasets have changed column names over time. This
    adapter accepts both the newer ESPN-style columns and several legacy loader
    names, then exposes the columns stat_processing.py expects.
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

    # Schedule metadata is authoritative for game-level ids/final scores.
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

    # ------------------------------------------------------------------
    # Possession-team identity
    # ------------------------------------------------------------------
    # Newer ESPN-derived SportsDataverse PBP already exposes numeric team
    # ids at play start. Prefer those over fuzzy/name-based mapping.
    #
    # Legacy cfbfastR/CFBD PBP instead exposes character pos_team and
    # def_pos_team values, so we retain the name-mapping fallback below.

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
        pbp["pos_team"] = pd.to_numeric(pbp[direct_pos_id], errors="coerce")
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
        pbp["def_pos_team"] = pd.to_numeric(pbp[direct_def_id], errors="coerce")
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

    # If the direct numeric ids were not available (legacy CFBD/cfbfastR),
    # map team names/abbreviations to the schedule's ESPN numeric team ids.
    _map_team_values_to_ids(pbp, "pos_team")
    _map_team_values_to_ids(pbp, "def_pos_team")

    # ESPN PBP also contains an is_home/start.is_home possession flag.
    # Use it as a final deterministic fallback for any rows whose team name
    # could not be mapped exactly.
    possession_is_home_col = _first_existing(
        pbp,
        ["start.is_home", "is_home"],
    )
    if possession_is_home_col is not None:
        possession_is_home = _coerce_bool_series(pbp[possession_is_home_col])

        pos_missing = pbp["pos_team"].isna()
        pbp.loc[pos_missing & possession_is_home, "pos_team"] = pbp.loc[
            pos_missing & possession_is_home, "homeTeamId"
        ]
        pbp.loc[pos_missing & ~possession_is_home, "pos_team"] = pbp.loc[
            pos_missing & ~possession_is_home, "awayTeamId"
        ]

        def_missing = pbp["def_pos_team"].isna()
        pbp.loc[def_missing & possession_is_home, "def_pos_team"] = pbp.loc[
            def_missing & possession_is_home, "awayTeamId"
        ]
        pbp.loc[def_missing & ~possession_is_home, "def_pos_team"] = pbp.loc[
            def_missing & ~possession_is_home, "homeTeamId"
        ]

    pbp["pos_team"] = pd.to_numeric(pbp["pos_team"], errors="coerce")
    pbp["def_pos_team"] = pd.to_numeric(pbp["def_pos_team"], errors="coerce")

    # If the name/id conversion fails badly, every historical feature row
    # will be zero. Fail early with useful context.
    pos_mapped = pbp["pos_team"].notna().mean() if len(pbp) else 0.0
    def_mapped = pbp["def_pos_team"].notna().mean() if len(pbp) else 0.0
    if pos_mapped < 0.50 or def_mapped < 0.50:
        raw_pos = _as_pandas(raw_pbp)
        examples = []
        for col in ["pos_team", "def_pos_team", "home", "away"]:
            if col in raw_pos.columns:
                vals = raw_pos[col].dropna().astype(str).unique()[:5].tolist()
                examples.append(f"{col}={vals}")
        raise ValueError(
            "Could not map CFB possession-team names to ESPN team ids. "
            f"Mapped pos_team={pos_mapped:.1%}, def_pos_team={def_mapped:.1%}. "
            + " | ".join(examples)
        )

    # Construct turnover flag if this version of the data does not expose one.
    if "turnover_vec" not in pbp.columns:
        interception = (
            pbp["int"].fillna(False).astype(bool)
            if "int" in pbp.columns
            else pd.Series(False, index=pbp.index)
        )
        fumble_lost = (
            pbp["fumble_lost"].fillna(False).astype(bool)
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
        pbp[column] = pd.to_numeric(pbp[column], errors="coerce")

    for column in ["pass", "rush", "sack", "turnover_vec"]:
        pbp[column] = _coerce_bool_series(pbp[column])

    if "penalty_no_play" in pbp.columns:
        pbp["penalty_no_play"] = pbp["penalty_no_play"].fillna(False).astype(bool)

    # A historical game is usable when the explicit completion flag is true OR
    # both final scores are present. This avoids throwing away valid historical
    # PBP because one release has a missing/stale status flag.
    score_completed = pbp["homeScore"].notna() & pbp["awayScore"].notna()

    if "status_type_completed" in pbp.columns:
        pbp["status_type_completed"] = (
            _coerce_bool_series(pbp["status_type_completed"]) | score_completed
        )
    else:
        pbp["status_type_completed"] = score_completed

    return pbp


# =============================================================================
# SportsDataverse loaders
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
    """Load the current SportsDataverse CFB PBP surface.

    Prefer load_cfb_pbp() because the dashboard may include the current season.
    Older package builds can fall back to load_cfb_pbp_r().
    """

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
def load_model_data(
    first_feature_season: int,
    target_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one extra season so Week 1 can use prior-season history."""

    seasons = list(range(first_feature_season - 1, target_season + 1))

    raw_schedule = _load_cfb_schedule(seasons)
    schedule = normalize_schedule(raw_schedule)

    raw_pbp = _load_cfb_pbp(seasons)
    pbp = normalize_pbp_for_stats(raw_pbp, schedule)

    return pbp, schedule


# =============================================================================
# Feature engineering using stat_processing.py
# =============================================================================


@dataclass
class FeatureContext:
    pbp: pd.DataFrame
    pbp_by_game: pd.DataFrame
    team_game_ids: dict[int, list[int]]
    offense_cache: dict[tuple[int, int, int, int], dict[str, float]]
    defense_cache: dict[tuple[int, int, int, int], dict[str, float]]


def make_feature_context(pbp: pd.DataFrame) -> FeatureContext:
    game_teams = (
        pbp[["game_id", "homeTeamId", "awayTeamId"]]
        .dropna()
        .drop_duplicates("game_id")
        .copy()
    )
    game_teams[["game_id", "homeTeamId", "awayTeamId"]] = game_teams[
        ["game_id", "homeTeamId", "awayTeamId"]
    ].astype(int)

    team_game_ids: dict[int, list[int]] = {}
    for row in game_teams.itertuples(index=False):
        team_game_ids.setdefault(int(row.homeTeamId), []).append(int(row.game_id))
        team_game_ids.setdefault(int(row.awayTeamId), []).append(int(row.game_id))

    return FeatureContext(
        pbp=pbp,
        pbp_by_game=pbp.set_index("game_id", drop=False).sort_index(),
        team_game_ids=team_game_ids,
        offense_cache={},
        defense_cache={},
    )


def team_pbp_slice(context: FeatureContext, team_id: int) -> pd.DataFrame:
    game_ids = context.team_game_ids.get(int(team_id), [])
    if not game_ids:
        return context.pbp.iloc[0:0].copy()

    present = [gid for gid in game_ids if gid in context.pbp_by_game.index]
    if not present:
        return context.pbp.iloc[0:0].copy()

    # .loc against the game_id index avoids scanning the entire multi-season
    # play-by-play frame for every historical training row.
    return context.pbp_by_game.loc[present].copy().reset_index(drop=True)


def get_team_offense(
    context: FeatureContext,
    team_id: int,
    season: int,
    week: int,
    lookback_games: int,
) -> dict[str, float]:
    key = (int(team_id), int(season), int(week), int(lookback_games))
    if key not in context.offense_cache:
        context.offense_cache[key] = stat_processing.get_offensive_team_stats_past_n_games(
            team=int(team_id),
            n=lookback_games,
            pbp_data=team_pbp_slice(context, int(team_id)),
            before_season=int(season),
            before_week=int(week),
        )
    return context.offense_cache[key]


def get_team_defense(
    context: FeatureContext,
    team_id: int,
    season: int,
    week: int,
    lookback_games: int,
) -> dict[str, float]:
    key = (int(team_id), int(season), int(week), int(lookback_games))
    if key not in context.defense_cache:
        context.defense_cache[key] = stat_processing.get_defensive_team_stats_past_n_games(
            team=int(team_id),
            n=lookback_games,
            pbp_data=team_pbp_slice(context, int(team_id)),
            before_season=int(season),
            before_week=int(week),
        )
    return context.defense_cache[key]


def build_feature_row(
    context: FeatureContext,
    offense_team: int,
    defense_team: int,
    is_home: bool,
    season: int,
    week: int,
    lookback_games: int,
) -> dict[str, float]:
    """Build exactly the X row requested by the model."""

    offense = get_team_offense(
        context,
        offense_team,
        season,
        week,
        lookback_games,
    )
    defense = get_team_defense(
        context,
        defense_team,
        season,
        week,
        lookback_games,
    )

    row = {
        "is_home": float(bool(is_home)),
        **offense,
        **defense,
    }

    return {column: float(row[column]) for column in FEATURE_COLUMNS}


def is_sec_game(row: pd.Series) -> bool:
    home_conf = str(row.get("home_conference", "")).strip().upper()
    away_conf = str(row.get("away_conference", "")).strip().upper()

    if home_conf == "SEC" or away_conf == "SEC":
        return True

    return int(row["home_id"]) in SEC_TEAM_IDS or int(row["away_id"]) in SEC_TEAM_IDS


def games_before_target(
    schedule: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    return schedule[
        (schedule["season"] < target_season)
        | ((schedule["season"] == target_season) & (schedule["week"] < target_week))
    ].copy()


def build_historical_training_rows(
    first_training_season: int,
    target_season: int,
    target_week: int,
    lookback_games: int,
    progress_callback=None,
    exclude_game_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Create two leakage-safe score rows per historical SEC-related game."""

    pbp, schedule = load_model_data(first_training_season, target_season)
    context = make_feature_context(pbp)

    eligible = games_before_target(schedule, target_season, target_week)

    diagnostic_counts = {
        "schedule_rows_before_target": int(len(eligible)),
    }

    # Scores are authoritative evidence that a historical game was actually
    # played. Keep the explicit completion flag too, but do not let a stale
    # status flag erase a game whose final scores are present.
    score_completed = eligible["home_score"].notna() & eligible["away_score"].notna()
    completed_mask = eligible["completed"].fillna(False).astype(bool) | score_completed

    eligible = eligible[
        completed_mask
        & (eligible["season"] >= first_training_season)
        & score_completed
    ].copy()
    diagnostic_counts["completed_scored_rows"] = int(len(eligible))

    # Train in the same domain that the dashboard predicts: games involving a
    # current SEC program. Both teams receive a row, so non-SEC opponents are
    # represented in X/y as well.
    eligible = eligible[eligible.apply(is_sec_game, axis=1)]
    diagnostic_counts["sec_related_rows"] = int(len(eligible))

    pbp_game_ids = set(
        pd.to_numeric(pbp["game_id"], errors="coerce").dropna().astype(int)
    )
    eligible = eligible[eligible["game_id"].isin(pbp_game_ids)]
    diagnostic_counts["schedule_pbp_matched_rows"] = int(len(eligible))

    eligible = eligible.sort_values(["season", "week", "start_date", "game_id"])

    if exclude_game_ids:
        eligible = eligible[~eligible["game_id"].isin(exclude_game_ids)].copy()

    rows: list[dict] = []

    total_games = len(eligible)

    if progress_callback is not None:
        progress_callback(
            5,
            f"Historical features: {total_games:,} eligible games found.",
        )

    for game_number, game in enumerate(
        eligible.itertuples(index=False),
        start=1,
    ):
        home_id = int(game.home_id)
        away_id = int(game.away_id)
        season = int(game.season)
        week = int(game.week)

        home_features = build_feature_row(
            context,
            offense_team=home_id,
            defense_team=away_id,
            is_home=True,
            season=season,
            week=week,
            lookback_games=lookback_games,
        )
        away_features = build_feature_row(
            context,
            offense_team=away_id,
            defense_team=home_id,
            is_home=False,
            season=season,
            week=week,
            lookback_games=lookback_games,
        )

        # Rows with absolutely no historical signal usually mean the team has
        # no prior-season/current-season PBP in the loaded window.
        home_signal = sum(abs(home_features[c]) for c in FEATURE_COLUMNS if c != "is_home")
        away_signal = sum(abs(away_features[c]) for c in FEATURE_COLUMNS if c != "is_home")

        if home_signal > 0:
            rows.append(
                {
                    "game_id": int(game.game_id),
                    "season": season,
                    "week": week,
                    "start_date": game.start_date,
                    "team_id": home_id,
                    "opponent_id": away_id,
                    "team_name": game.home_team,
                    "opponent_name": game.away_team,
                    "actual_points": float(game.home_score),
                    **home_features,
                }
            )

        if away_signal > 0:
            rows.append(
                {
                    "game_id": int(game.game_id),
                    "season": season,
                    "week": week,
                    "start_date": game.start_date,
                    "team_id": away_id,
                    "opponent_id": home_id,
                    "team_name": game.away_team,
                    "opponent_name": game.home_team,
                    "actual_points": float(game.away_score),
                    **away_features,
                }
            )

        if progress_callback is not None and total_games > 0:
            percent = 5 + int(95 * game_number / total_games)
            progress_callback(
                min(percent, 100),
                (
                    f"Historical features: {game_number:,}/{total_games:,} games "
                    f"processed • {len(rows):,} scoring rows kept"
                ),
            )

    if not rows:
        # Give enough information to identify whether the schedule filter,
        # SEC filter, PBP join, or feature builder is responsible.
        diagnostic_counts["pbp_rows"] = int(len(pbp))
        diagnostic_counts["pbp_unique_games"] = int(
            pd.to_numeric(pbp["game_id"], errors="coerce").dropna().nunique()
        )
        diagnostic_counts["eligible_games_entering_feature_loop"] = int(len(eligible))

        sample_feature_diagnostics = []
        for game in eligible.head(3).itertuples(index=False):
            home_features = build_feature_row(
                context,
                offense_team=int(game.home_id),
                defense_team=int(game.away_id),
                is_home=True,
                season=int(game.season),
                week=int(game.week),
                lookback_games=lookback_games,
            )
            away_features = build_feature_row(
                context,
                offense_team=int(game.away_id),
                defense_team=int(game.home_id),
                is_home=False,
                season=int(game.season),
                week=int(game.week),
                lookback_games=lookback_games,
            )
            home_signal = sum(
                abs(home_features[c]) for c in FEATURE_COLUMNS if c != "is_home"
            )
            away_signal = sum(
                abs(away_features[c]) for c in FEATURE_COLUMNS if c != "is_home"
            )
            home_debug = stat_processing.debug_team_history(
                team=int(game.home_id),
                n=lookback_games,
                pbp_data=team_pbp_slice(context, int(game.home_id)),
                before_season=int(game.season),
                before_week=int(game.week),
            )
            away_debug = stat_processing.debug_team_history(
                team=int(game.away_id),
                n=lookback_games,
                pbp_data=team_pbp_slice(context, int(game.away_id)),
                before_season=int(game.season),
                before_week=int(game.week),
            )

            sample_feature_diagnostics.append(
                f"game={int(game.game_id)} "
                f"{game.away_team}@{game.home_team}: "
                f"away_signal={away_signal:.4f}, home_signal={home_signal:.4f}; "
                f"away_debug={away_debug}; home_debug={home_debug}"
            )

        details = ", ".join(
            f"{key}={value}" for key, value in diagnostic_counts.items()
        )
        samples = (
            " | samples: " + "; ".join(sample_feature_diagnostics)
            if sample_feature_diagnostics
            else ""
        )
        raise ValueError(
            "No historical training rows could be created. "
            + details
            + samples
        )

    return pd.DataFrame(rows).sort_values(
        ["season", "week", "start_date", "game_id", "is_home"]
    ).reset_index(drop=True)


# =============================================================================
# Persistent incremental historical feature store
# =============================================================================

FEATURE_STORE_DIR = Path(__file__).resolve().parent / "data" / "feature_store"


def historical_feature_store_path(
    first_training_season: int,
    target_season: int,
    lookback_games: int,
) -> Path:
    return FEATURE_STORE_DIR / (
        f"historical_features_{first_training_season}_{target_season}_"
        f"lb{lookback_games}.parquet"
    )


def rows_before_target(
    rows: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    mask = (
        (rows["season"] < int(target_season))
        | ((rows["season"] == int(target_season)) & (rows["week"] < int(target_week)))
    )
    return rows.loc[mask].copy().sort_values(
        ["season", "week", "start_date", "game_id", "is_home"]
    ).reset_index(drop=True)


def load_incremental_historical_features(
    first_training_season: int,
    target_season: int,
    target_week: int,
    lookback_games: int,
    progress_callback=None,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, Path, int]:
    """Load shared historical features from Parquet and append missing games."""
    store_path = historical_feature_store_path(
        int(first_training_season), int(target_season), int(lookback_games)
    )
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild and store_path.exists():
        store_path.unlink()

    if store_path.exists():
        cached = pd.read_parquet(store_path)
        if not cached.empty:
            cached["game_id"] = pd.to_numeric(cached["game_id"], errors="coerce")
            cached = cached.dropna(subset=["game_id"]).copy()
            cached["game_id"] = cached["game_id"].astype(int)
        if progress_callback is not None:
            progress_callback(10, f"Historical features: loaded {len(cached):,} persisted rows from Parquet.")
    else:
        cached = pd.DataFrame()

    cached_game_ids = (
        set(cached["game_id"].astype(int).unique())
        if not cached.empty and "game_id" in cached.columns
        else set()
    )

    pbp, schedule = load_model_data(int(first_training_season), int(target_season))
    needed = games_before_target(schedule, int(target_season), int(target_week))
    score_completed = needed["home_score"].notna() & needed["away_score"].notna()
    completed_mask = needed["completed"].fillna(False).astype(bool) | score_completed
    needed = needed[
        completed_mask
        & (needed["season"] >= int(first_training_season))
        & score_completed
    ].copy()
    needed = needed[needed.apply(is_sec_game, axis=1)]
    pbp_game_ids = set(pd.to_numeric(pbp["game_id"], errors="coerce").dropna().astype(int))
    needed = needed[needed["game_id"].isin(pbp_game_ids)]
    needed_ids = set(pd.to_numeric(needed["game_id"], errors="coerce").dropna().astype(int))
    missing_ids = needed_ids - cached_game_ids

    appended_rows = 0
    if missing_ids:
        if progress_callback is not None:
            progress_callback(15, f"Historical features: {len(missing_ids):,} new games need feature generation.")
        new_rows = build_historical_training_rows(
            first_training_season=int(first_training_season),
            target_season=int(target_season),
            target_week=int(target_week),
            lookback_games=int(lookback_games),
            progress_callback=progress_callback,
            exclude_game_ids=cached_game_ids,
        )
        appended_rows = len(new_rows)
        persisted = new_rows.copy() if cached.empty else pd.concat([cached, new_rows], ignore_index=True)
        persisted = persisted.drop_duplicates(
            subset=["game_id", "team_id", "is_home"], keep="last"
        ).sort_values(
            ["season", "week", "start_date", "game_id", "is_home"]
        ).reset_index(drop=True)
        temp_path = store_path.with_suffix(".tmp.parquet")
        persisted.to_parquet(temp_path, index=False)
        temp_path.replace(store_path)
        cached = persisted
    elif progress_callback is not None:
        progress_callback(100, "Historical features: Parquet store is already current.")

    training_rows = rows_before_target(cached, int(target_season), int(target_week))
    if training_rows.empty:
        raise ValueError("No historical training rows are available in the feature store.")
    return training_rows, store_path, appended_rows


# =============================================================================
# Regression model + out-of-sample residuals
# =============================================================================


@dataclass
class ModelBundle:
    model: RandomForestRegressor
    validation_results: pd.DataFrame
    paired_residuals: pd.DataFrame
    all_residuals: np.ndarray
    mae: float
    rmse: float
    r2: float
    train_rows: int
    validation_rows: int


def train_and_validate_model(
    training_rows: pd.DataFrame,
    progress_callback=None,
) -> ModelBundle:
    """Chronological game-level split, validation, then refit on all history."""

    def update_progress(value: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    update_progress(5, "Preparing chronological train/validation split...")

    unique_games = (
        training_rows[["game_id", "season", "week", "start_date"]]
        .drop_duplicates("game_id")
        .sort_values(["season", "week", "start_date", "game_id"])
    )

    if len(unique_games) < 40:
        raise ValueError(
            f"Only {len(unique_games)} historical games are available; "
            "increase the training window before fitting the model."
        )

    n_validation_games = max(20, int(round(len(unique_games) * VALIDATION_FRACTION)))
    n_validation_games = min(n_validation_games, len(unique_games) - 20)

    validation_game_ids = set(unique_games.tail(n_validation_games)["game_id"])

    train_mask = ~training_rows["game_id"].isin(validation_game_ids)
    validation_mask = training_rows["game_id"].isin(validation_game_ids)

    train = training_rows[train_mask].copy()
    validation = training_rows[validation_mask].copy()

    X_train = train[FEATURE_COLUMNS].astype(float)
    y_train = train["actual_points"].astype(float)
    X_validation = validation[FEATURE_COLUMNS].astype(float)
    y_validation = validation["actual_points"].astype(float)

    update_progress(
        20,
        f"Training validation Random Forest on {len(train):,} scoring rows...",
    )

    validation_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features=0.80,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    validation_model.fit(X_train, y_train)

    update_progress(
        55,
        f"Validating on {len(validation):,} held-out scoring rows...",
    )
    validation_predictions = validation_model.predict(X_validation)

    validation_results = validation[
        [
            "game_id",
            "season",
            "week",
            "team_id",
            "opponent_id",
            "team_name",
            "opponent_name",
            "is_home",
            "actual_points",
        ]
    ].copy()
    validation_results["predicted_points"] = validation_predictions
    validation_results["residual"] = (
        validation_results["actual_points"] - validation_results["predicted_points"]
    )

    mae = float(mean_absolute_error(y_validation, validation_predictions))
    rmse = float(np.sqrt(mean_squared_error(y_validation, validation_predictions)))
    r2 = float(r2_score(y_validation, validation_predictions))

    update_progress(
        70,
        f"Validation complete • MAE {mae:.2f} pts • RMSE {rmse:.2f} pts",
    )

    # Pair the home and away residual from the SAME held-out historical game.
    paired = validation_results.pivot_table(
        index="game_id",
        columns="is_home",
        values="residual",
        aggfunc="first",
    )
    paired = paired.rename(columns={0.0: "away_residual", 1.0: "home_residual"})

    if "home_residual" in paired.columns and "away_residual" in paired.columns:
        paired = paired[["home_residual", "away_residual"]].dropna().reset_index()
    else:
        paired = pd.DataFrame(columns=["game_id", "home_residual", "away_residual"])

    update_progress(78, "Building held-out residual distributions...")

    # Refit the production model on EVERY historical row after validation.
    update_progress(
        82,
        f"Training final production model on all {len(training_rows):,} scoring rows...",
    )

    final_model = RandomForestRegressor(
        n_estimators=700,
        min_samples_leaf=4,
        max_features=0.80,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    final_model.fit(
        training_rows[FEATURE_COLUMNS].astype(float),
        training_rows["actual_points"].astype(float),
    )

    update_progress(100, "Model training and validation complete.")

    return ModelBundle(
        model=final_model,
        validation_results=validation_results,
        paired_residuals=paired,
        all_residuals=validation_results["residual"].to_numpy(dtype=float),
        mae=mae,
        rmse=rmse,
        r2=r2,
        train_rows=len(train),
        validation_rows=len(validation),
    )


# =============================================================================
# Current-week SEC predictions
# =============================================================================


def get_default_week(schedule: pd.DataFrame, season: int) -> int:
    games = schedule[schedule["season"] == season].dropna(subset=["week"]).copy()
    if games.empty:
        raise ValueError(f"No schedule data is available for {season}.")

    now = pd.Timestamp.now(tz="UTC")

    if games["start_date"].notna().any():
        week_dates = games.groupby("week")["start_date"].agg(["min", "max"])

        for week, row in week_dates.iterrows():
            if pd.notna(row["min"]) and pd.notna(row["max"]):
                if row["min"] - pd.Timedelta(days=2) <= now <= row["max"] + pd.Timedelta(days=2):
                    return int(week)

        midpoint = week_dates["min"] + (week_dates["max"] - week_dates["min"]) / 2
        return int((midpoint - now).abs().idxmin())

    return int(games["week"].max())


def get_sec_week_games(schedule: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = schedule[(schedule["season"] == season) & (schedule["week"] == week)].copy()
    if games.empty:
        return games
    mask = games.apply(is_sec_game, axis=1)
    return games[mask].sort_values(["start_date", "game_id"]).reset_index(drop=True)


def add_current_predictions(
    games: pd.DataFrame,
    pbp: pd.DataFrame,
    model: RandomForestRegressor,
    season: int,
    week: int,
    lookback_games: int,
) -> pd.DataFrame:
    """Build matchup features, then score every team in one model.predict call."""
    context = make_feature_context(pbp)
    pending = []
    feature_rows = []

    for game in games.itertuples(index=False):
        home_features = build_feature_row(
            context,
            offense_team=int(game.home_id),
            defense_team=int(game.away_id),
            is_home=True,
            season=season,
            week=week,
            lookback_games=lookback_games,
        )
        away_features = build_feature_row(
            context,
            offense_team=int(game.away_id),
            defense_team=int(game.home_id),
            is_home=False,
            season=season,
            week=week,
            lookback_games=lookback_games,
        )

        pending.append((game._asdict(), home_features, away_features))
        feature_rows.extend((home_features, away_features))

    if not pending:
        return pd.DataFrame()

    # sklearn RandomForest remains CPU-based, but one batched call avoids many
    # tiny DataFrame allocations and predict() calls.
    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    predicted = np.maximum(model.predict(X).astype(float), 0.0).reshape(-1, 2)

    output_rows = []
    for index, (game_dict, home_features, away_features) in enumerate(pending):
        output_rows.append(
            {
                **game_dict,
                "expected_home_score": float(predicted[index, 0]),
                "expected_away_score": float(predicted[index, 1]),
                "home_features": home_features,
                "away_features": away_features,
            }
        )

    return pd.DataFrame(output_rows)


# =============================================================================
# Empirical-residual Monte Carlo (PyTorch/ROCm accelerated when available)
# =============================================================================


def _torch_residuals(bundle: ModelBundle):
    """Cache tiny validation-residual tensors on the active accelerator."""
    if not GPU_ACCELERATION_AVAILABLE:
        return None, None

    device_key = str(TORCH_DEVICE)
    cached_key = getattr(bundle, "_torch_residual_device", None)
    paired = getattr(bundle, "_torch_paired_residuals", None)
    all_residuals = getattr(bundle, "_torch_all_residuals", None)

    if cached_key == device_key and paired is not None and all_residuals is not None:
        return paired, all_residuals

    if len(bundle.paired_residuals):
        paired_np = bundle.paired_residuals[
            ["home_residual", "away_residual"]
        ].to_numpy(dtype=np.float32, copy=True)
    else:
        paired_np = np.empty((0, 2), dtype=np.float32)

    all_np = np.asarray(bundle.all_residuals, dtype=np.float32)

    paired = torch.as_tensor(paired_np, device=TORCH_DEVICE)
    all_residuals = torch.as_tensor(all_np, device=TORCH_DEVICE)

    bundle._torch_residual_device = device_key
    bundle._torch_paired_residuals = paired
    bundle._torch_all_residuals = all_residuals
    return paired, all_residuals


def simulate_game_from_residuals(
    expected_home: float,
    expected_away: float,
    bundle: ModelBundle,
    n: int = MONTE_CARLO_SIMS,
    seed: int = RANDOM_SEED,
):
    """Simulate one game; returns GPU tensors on ROCm and NumPy arrays on CPU."""
    if n <= 0:
        raise ValueError("n must be positive")

    if GPU_ACCELERATION_AVAILABLE:
        paired, all_residuals = _torch_residuals(bundle)
        generator = torch.Generator(device=TORCH_DEVICE)
        generator.manual_seed(int(seed))

        if paired.shape[0] >= 20:
            sampled_indices = torch.randint(
                0, paired.shape[0], (int(n),),
                generator=generator, device=TORCH_DEVICE,
            )
            sampled = paired[sampled_indices]
            home_error = sampled[:, 0]
            away_error = sampled[:, 1]
        else:
            if all_residuals.numel() == 0:
                raise ValueError("No validation residuals are available for simulation.")
            home_idx = torch.randint(
                0, all_residuals.numel(), (int(n),),
                generator=generator, device=TORCH_DEVICE,
            )
            away_idx = torch.randint(
                0, all_residuals.numel(), (int(n),),
                generator=generator, device=TORCH_DEVICE,
            )
            home_error = all_residuals[home_idx]
            away_error = all_residuals[away_idx]

        home_scores = torch.round(float(expected_home) + home_error).clamp_min_(0).to(torch.int32)
        away_scores = torch.round(float(expected_away) + away_error).clamp_min_(0).to(torch.int32)
        return home_scores, away_scores

    # CPU fallback preserves the original NumPy implementation.
    rng = np.random.default_rng(seed)
    if len(bundle.paired_residuals) >= 20:
        sampled_indices = rng.integers(0, len(bundle.paired_residuals), size=n)
        sampled = bundle.paired_residuals.iloc[sampled_indices]
        home_error = sampled["home_residual"].to_numpy(dtype=float)
        away_error = sampled["away_residual"].to_numpy(dtype=float)
    else:
        if len(bundle.all_residuals) == 0:
            raise ValueError("No validation residuals are available for simulation.")
        home_error = rng.choice(bundle.all_residuals, size=n, replace=True)
        away_error = rng.choice(bundle.all_residuals, size=n, replace=True)

    home_scores = np.clip(np.rint(expected_home + home_error), 0, None).astype(int)
    away_scores = np.clip(np.rint(expected_away + away_error), 0, None).astype(int)
    return home_scores, away_scores


def simulate_games_from_residuals(
    expected_home,
    expected_away,
    bundle: ModelBundle,
    n: int = MONTE_CARLO_SIMS,
    seed: int = RANDOM_SEED,
):
    """Batch-simulate many games at once. Shape is (games, simulations)."""
    home_expected_np = np.asarray(expected_home, dtype=np.float32).reshape(-1)
    away_expected_np = np.asarray(expected_away, dtype=np.float32).reshape(-1)
    if home_expected_np.shape != away_expected_np.shape:
        raise ValueError("expected_home and expected_away must have matching shapes")
    if n <= 0:
        raise ValueError("n must be positive")

    game_count = int(home_expected_np.size)
    if game_count == 0:
        if GPU_ACCELERATION_AVAILABLE:
            empty = torch.empty((0, int(n)), dtype=torch.int32, device=TORCH_DEVICE)
            return empty, empty.clone()
        empty = np.empty((0, int(n)), dtype=int)
        return empty, empty.copy()

    if GPU_ACCELERATION_AVAILABLE:
        paired, all_residuals = _torch_residuals(bundle)
        generator = torch.Generator(device=TORCH_DEVICE)
        generator.manual_seed(int(seed))
        shape = (game_count, int(n))

        if paired.shape[0] >= 20:
            sampled_indices = torch.randint(
                0, paired.shape[0], shape,
                generator=generator, device=TORCH_DEVICE,
            )
            sampled = paired[sampled_indices]
            home_error = sampled[..., 0]
            away_error = sampled[..., 1]
        else:
            if all_residuals.numel() == 0:
                raise ValueError("No validation residuals are available for simulation.")
            home_idx = torch.randint(
                0, all_residuals.numel(), shape,
                generator=generator, device=TORCH_DEVICE,
            )
            away_idx = torch.randint(
                0, all_residuals.numel(), shape,
                generator=generator, device=TORCH_DEVICE,
            )
            home_error = all_residuals[home_idx]
            away_error = all_residuals[away_idx]

        home_expected_t = torch.as_tensor(home_expected_np, device=TORCH_DEVICE).unsqueeze(1)
        away_expected_t = torch.as_tensor(away_expected_np, device=TORCH_DEVICE).unsqueeze(1)
        home_scores = torch.round(home_expected_t + home_error).clamp_min_(0).to(torch.int32)
        away_scores = torch.round(away_expected_t + away_error).clamp_min_(0).to(torch.int32)
        return home_scores, away_scores

    rng = np.random.default_rng(seed)
    shape = (game_count, int(n))
    if len(bundle.paired_residuals) >= 20:
        paired_np = bundle.paired_residuals[["home_residual", "away_residual"]].to_numpy(dtype=float)
        idx = rng.integers(0, len(paired_np), size=shape)
        sampled = paired_np[idx]
        home_error = sampled[..., 0]
        away_error = sampled[..., 1]
    else:
        if len(bundle.all_residuals) == 0:
            raise ValueError("No validation residuals are available for simulation.")
        home_error = rng.choice(bundle.all_residuals, size=shape, replace=True)
        away_error = rng.choice(bundle.all_residuals, size=shape, replace=True)

    home_scores = np.clip(np.rint(home_expected_np[:, None] + home_error), 0, None).astype(int)
    away_scores = np.clip(np.rint(away_expected_np[:, None] + away_error), 0, None).astype(int)
    return home_scores, away_scores


def _is_torch_tensor(value) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def game_probabilities(home, away) -> tuple[float, float, float]:
    if _is_torch_tensor(home):
        home_win = (home > away).float().mean().item()
        away_win = (away > home).float().mean().item()
        tie = (home == away).float().mean().item()
        return float(home_win), float(away_win), float(tie)

    home = np.asarray(home)
    away = np.asarray(away)
    return (
        float(np.mean(home > away)),
        float(np.mean(away > home)),
        float(np.mean(home == away)),
    )


def game_probabilities_batch(home, away):
    """Return one home/away/tie probability per row of batched simulations."""
    if _is_torch_tensor(home):
        dims = 1 if home.ndim > 1 else 0
        home_win = (home > away).float().mean(dim=dims)
        away_win = (away > home).float().mean(dim=dims)
        tie = (home == away).float().mean(dim=dims)
        return (
            home_win.detach().cpu().numpy(),
            away_win.detach().cpu().numpy(),
            tie.detach().cpu().numpy(),
        )

    home = np.asarray(home)
    away = np.asarray(away)
    axis = 1 if home.ndim > 1 else 0
    return (
        np.mean(home > away, axis=axis),
        np.mean(away > home, axis=axis),
        np.mean(home == away, axis=axis),
    )


def score_distribution(scores) -> pd.DataFrame:
    if _is_torch_tensor(scores):
        flat = scores.reshape(-1).to(torch.int64)
        counts = torch.bincount(flat)
        present = counts > 0
        values = torch.arange(counts.numel(), device=scores.device)[present]
        probabilities = counts[present].float() / flat.numel()
        return pd.DataFrame(
            {
                "Score": values.detach().cpu().numpy(),
                "Probability": probabilities.detach().cpu().numpy(),
            }
        )

    values, counts = np.unique(np.asarray(scores), return_counts=True)
    probabilities = counts / len(scores)
    return pd.DataFrame({"Score": values, "Probability": probabilities})


def likely_final_scores(home, away, top_n: int = 12) -> pd.DataFrame:
    if _is_torch_tensor(home):
        home_flat = home.reshape(-1).to(torch.int64)
        away_flat = away.reshape(-1).to(torch.int64)
        max_score = int(torch.maximum(home_flat.max(), away_flat.max()).item()) + 1
        encoded = away_flat * max_score + home_flat
        values, counts = torch.unique(encoded, return_counts=True)
        probabilities = counts.float() / encoded.numel()
        k = min(int(top_n), int(probabilities.numel()))
        top_probability, top_indices = torch.topk(probabilities, k=k)
        selected = values[top_indices]
        return pd.DataFrame(
            {
                "Away": (selected // max_score).detach().cpu().numpy(),
                "Home": (selected % max_score).detach().cpu().numpy(),
                "Probability": top_probability.detach().cpu().numpy(),
            }
        )

    outcomes = pd.DataFrame({"Away": np.asarray(away), "Home": np.asarray(home)})
    result = outcomes.value_counts().rename("Count").reset_index().head(top_n)
    result["Probability"] = result["Count"] / len(outcomes)
    return result.drop(columns="Count")


def score_percentile(scores, percentile: float) -> float:
    """Tensor-aware percentile helper used by Streamlit views."""
    if _is_torch_tensor(scores):
        q = float(percentile) / 100.0
        return float(torch.quantile(scores.float(), q).item())
    return float(np.percentile(np.asarray(scores), percentile))
