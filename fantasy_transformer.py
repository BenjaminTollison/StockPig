"""
fantasy_transformer.py

Player-outcome architecture for StockPig fantasy football.

The Transformer is scoring-agnostic. It learns NEXT-SEASON football outcome
Distributions from historical NFL weekly data. Sleeper scoring is applied only
after the football outcomes are projected.

Current preseason flow:
    SportsDataverse / nflverse weekly history
        -> Transformer encoder
        -> season outcome distributions
        -> Fantasy Projection Engine (Sleeper scoring)
        -> VORP / draft rankings

The same outcome schema is used for rookies. Because rookies have no NFL weekly
history, a separate draft/combine random-forest prior predicts their outcome
Distributions. That keeps veterans and rookies on the same football-stat scale.

A later weekly model can reuse the encoder and add opponent/schedule context
before the outcome heads.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
POSITION_TO_ID = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

# The Transformer sees only football information. League-specific fantasy
# points are deliberately excluded from the sequence.
WEEKLY_FEATURES = [
    "week",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "passing_air_yards",
    "passing_epa",
    "dakota",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "fumbles_lost",
]

# The model outputs an independent marginal distribution for each quantity.
# These are season totals in the current draft/preseason model.
OUTCOME_TARGETS = [
    "games",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "fumbles_lost",
]
OUTCOME_TO_INDEX = {name: idx for idx, name in enumerate(OUTCOME_TARGETS)}

# Only meaningful heads are trained for each position. This prevents the model
# from being rewarded for learning that, for example, WR passing yards are zero.
POSITION_OUTCOMES = {
    "QB": {
        "games",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "fumbles_lost",
    },
    "RB": {
        "games",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "fumbles_lost",
    },
    "WR": {
        "games",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "fumbles_lost",
    },
    "TE": {
        "games",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "fumbles_lost",
    },
}

STATIC_FEATURES = [
    "age_at_target",
    "experience_at_target",
    "height",
    "weight",
    "draft_round",
    "draft_pick",
]

ROOKIE_NUMERIC_FEATURES = [
    "draft_round",
    "draft_pick",
    "height",
    "weight",
    "forty",
    "bench",
    "vertical",
    "broad_jump",
    "cone",
    "shuttle",
]


@dataclass
class FantasyScoring:
    """League scoring used AFTER football outcomes are projected."""

    reception: float = 1.0
    passing_yard: float = 0.04
    passing_td: float = 4.0
    interception: float = -2.0
    rushing_yard: float = 0.10
    rushing_td: float = 6.0
    receiving_yard: float = 0.10
    receiving_td: float = 6.0
    passing_2pt: float = 2.0
    rushing_2pt: float = 2.0
    receiving_2pt: float = 2.0
    fumble_lost: float = -2.0
    raw_sleeper_settings: dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_sleeper_settings(cls, settings: dict | None) -> "FantasyScoring":
        settings = settings or {}

        def value(key: str, default: float) -> float:
            raw = settings.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(default)

        return cls(
            reception=value("rec", 1.0),
            passing_yard=value("pass_yd", 0.04),
            passing_td=value("pass_td", 4.0),
            interception=value("pass_int", -2.0),
            rushing_yard=value("rush_yd", 0.10),
            rushing_td=value("rush_td", 6.0),
            receiving_yard=value("rec_yd", 0.10),
            receiving_td=value("rec_td", 6.0),
            passing_2pt=value("pass_2pt", 2.0),
            rushing_2pt=value("rush_2pt", 2.0),
            receiving_2pt=value("rec_2pt", 2.0),
            fumble_lost=value("fum_lost", -2.0),
            raw_sleeper_settings={
                str(k): float(v)
                for k, v in settings.items()
                if isinstance(v, (int, float))
            },
        )

    def unsupported_offensive_settings(self) -> dict[str, float]:
        supported = {
            "rec",
            "pass_yd",
            "pass_td",
            "pass_int",
            "rush_yd",
            "rush_td",
            "rec_yd",
            "rec_td",
            "pass_2pt",
            "rush_2pt",
            "rec_2pt",
            "fum_lost",
        }
        offensive_prefixes = (
            "pass_",
            "rush_",
            "rec_",
            "bonus_pass",
            "bonus_rush",
            "bonus_rec",
            "fum",
        )
        return {
            key: value
            for key, value in self.raw_sleeper_settings.items()
            if key not in supported
            and value != 0
            and key.startswith(offensive_prefixes)
        }


@dataclass
class TrainConfig:
    start_season: int = 2006
    sequence_length: int = 34
    min_history_games: int = 8
    min_target_games: int = 2
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 384
    dropout: float = 0.10
    batch_size: int = 128
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    random_seed: int = 42
    rookie_trees: int = 500
    max_regular_season_games: int = 17
    # Temporary draft-value defaults. The league-context phase should replace
    # these with values derived from actual Sleeper roster rules.
    replacement_rank_qb: int = 12
    replacement_rank_rb: int = 30
    replacement_rank_wr: int = 36
    replacement_rank_te: int = 12


@dataclass
class PreparedSample:
    player_id: str
    player_name: str
    position: str
    target_season: int
    sequence: np.ndarray
    static: np.ndarray
    target_outcomes: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_name(value: object) -> str:
    s = "" if pd.isna(value) else str(value)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _normalize_player_id(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return value


def _normalize_sleeper_id(value: object) -> str:
    """Normalize Sleeper IDs so CSV numeric coercion does not change identity."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def position_outcome_mask(position: str) -> np.ndarray:
    valid = POSITION_OUTCOMES.get(str(position).upper(), set())
    return np.array(
        [1.0 if target in valid else 0.0 for target in OUTCOME_TARGETS],
        dtype=np.float32,
    )


def fantasy_scoring_coefficients(scoring: FantasyScoring) -> np.ndarray:
    """Linear coefficients that convert football outcomes into fantasy points."""
    coefficients = {
        "games": 0.0,
        "completions": 0.0,
        "attempts": 0.0,
        "passing_yards": scoring.passing_yard,
        "passing_tds": scoring.passing_td,
        "interceptions": scoring.interception,
        "carries": 0.0,
        "rushing_yards": scoring.rushing_yard,
        "rushing_tds": scoring.rushing_td,
        "targets": 0.0,
        "receptions": scoring.reception,
        "receiving_yards": scoring.receiving_yard,
        "receiving_tds": scoring.receiving_td,
        "passing_2pt_conversions": scoring.passing_2pt,
        "rushing_2pt_conversions": scoring.rushing_2pt,
        "receiving_2pt_conversions": scoring.receiving_2pt,
        "fumbles_lost": scoring.fumble_lost,
    }
    return np.array([coefficients[name] for name in OUTCOME_TARGETS], dtype=np.float64)


def score_outcome_vector(outcomes: np.ndarray, scoring: FantasyScoring) -> np.ndarray:
    """Score one vector or a matrix of football outcomes."""
    coefficients = fantasy_scoring_coefficients(scoring)
    return np.asarray(outcomes, dtype=np.float64) @ coefficients


def add_fantasy_points(stats: pd.DataFrame, scoring: FantasyScoring) -> pd.DataFrame:
    """
    Score observed SportsDataverse rows. Kept for diagnostics/backtests.

    The Transformer does NOT consume this column.
    """
    df = prepare_outcome_columns(stats)
    matrix = np.zeros((len(df), len(OUTCOME_TARGETS)), dtype=np.float64)
    for idx, target in enumerate(OUTCOME_TARGETS):
        if target == "games":
            matrix[:, idx] = 1.0
        else:
            matrix[:, idx] = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
    df["fantasy_points_model"] = score_outcome_vector(matrix, scoring)
    return df


def prepare_outcome_columns(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    base = [target for target in OUTCOME_TARGETS if target not in {"games", "fumbles_lost"}]
    fumble_fields = [
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
    ]
    df = _to_numeric(df, base + fumble_fields)
    df["fumbles_lost"] = (
        df["sack_fumbles_lost"].fillna(0.0)
        + df["rushing_fumbles_lost"].fillna(0.0)
        + df["receiving_fumbles_lost"].fillna(0.0)
    )
    return df


def load_sportsdataverse(
    start_season: int,
    end_season: int,
    source: str = "nflverse",
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw football data. Fantasy scoring is intentionally absent here."""
    if progress:
        progress("Loading SportsDataverse weekly NFL player stats...")

    from sportsdataverse.nfl import (
        load_nfl_combine,
        load_nfl_draft_picks,
        load_nfl_player_stats,
        load_nfl_players,
    )

    stats = load_nfl_player_stats(return_as_pandas=True, source=source)
    season_values = pd.to_numeric(stats["season"], errors="coerce")
    stats = stats[
        (season_values >= start_season)
        & (season_values <= end_season)
    ].copy()

    if progress:
        progress("Loading SportsDataverse player identity data...")
    players = load_nfl_players(return_as_pandas=True, source=source)

    if progress:
        progress("Loading NFL combine and draft-capital data for rookies...")
    combine = load_nfl_combine(return_as_pandas=True)
    draft = load_nfl_draft_picks(return_as_pandas=True)

    return stats, players, combine, draft


def prepare_weekly_stats(stats: pd.DataFrame) -> pd.DataFrame:
    df = prepare_outcome_columns(stats)

    if "player_id" not in df.columns:
        raise ValueError("SportsDataverse player stats are missing player_id.")

    df["player_id"] = df["player_id"].map(_normalize_player_id)
    df = df[df["player_id"].ne("")].copy()

    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().eq("REG")]

    if "position" not in df.columns:
        raise ValueError("SportsDataverse player stats are missing position.")

    df["position"] = df["position"].astype(str).str.upper()
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()

    numeric = list(set(WEEKLY_FEATURES + ["season", "week"] + OUTCOME_TARGETS))
    df = _to_numeric(df, numeric)

    df[WEEKLY_FEATURES] = (
        df[WEEKLY_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return df


def prepare_players(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    if "gsis_id" not in p.columns:
        raise ValueError("SportsDataverse player master is missing gsis_id.")

    p["gsis_id"] = p["gsis_id"].map(_normalize_player_id)
    p = p[p["gsis_id"].ne("")].copy()
    p["position"] = p.get("position", "").astype(str).str.upper()

    for col in [
        "height",
        "weight",
        "rookie_season",
        "draft_year",
        "draft_round",
        "draft_pick",
    ]:
        if col not in p.columns:
            p[col] = np.nan
        p[col] = pd.to_numeric(p[col], errors="coerce")

    if "birth_date" not in p.columns:
        p["birth_date"] = pd.NaT
    p["birth_date"] = pd.to_datetime(p["birth_date"], errors="coerce")

    if "display_name" not in p.columns:
        p["display_name"] = ""

    if "sleeper_id" not in p.columns:
        for candidate in ("sleeper_player_id", "sleeper", "sleeperId"):
            if candidate in p.columns:
                p["sleeper_id"] = p[candidate]
                break

    if "sleeper_id" not in p.columns:
        p["sleeper_id"] = ""
    p["sleeper_id"] = p["sleeper_id"].map(_normalize_sleeper_id)
    return p


def _static_vector(player_row: pd.Series, target_season: int) -> np.ndarray:
    birth = player_row.get("birth_date", pd.NaT)
    age = np.nan if pd.isna(birth) else target_season - float(birth.year) + 0.5

    rookie_season = player_row.get("rookie_season", np.nan)
    if pd.isna(rookie_season):
        rookie_season = player_row.get("draft_year", np.nan)

    experience = (
        max(0.0, float(target_season) - float(rookie_season))
        if not pd.isna(rookie_season)
        else np.nan
    )

    draft_round = player_row.get("draft_round", np.nan)
    draft_pick = player_row.get("draft_pick", np.nan)
    if pd.isna(draft_round):
        draft_round = 8.0
    if pd.isna(draft_pick):
        draft_pick = 300.0

    return np.array(
        [
            age,
            experience,
            player_row.get("height", np.nan),
            player_row.get("weight", np.nan),
            draft_round,
            draft_pick,
        ],
        dtype=np.float32,
    )


def _player_name_column(df: pd.DataFrame) -> str:
    for column in ("player_name", "player_display_name", "display_name"):
        if column in df.columns:
            return column
    raise ValueError("Player stats do not contain a player name column.")


def aggregate_player_season_outcomes(stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly rows into one football-outcome vector per player-season."""
    stats = prepare_weekly_stats(stats)
    name_col = _player_name_column(stats)

    sum_targets = [target for target in OUTCOME_TARGETS if target != "games"]
    aggregation = {target: (target, "sum") for target in sum_targets}
    aggregation["games"] = ("week", "nunique")

    grouped = (
        stats.groupby(
            ["player_id", name_col, "position", "season"],
            dropna=False,
        )
        .agg(**aggregation)
        .reset_index()
        .rename(columns={name_col: "player_name"})
    )
    return grouped


def build_veteran_training_samples(
    stats: pd.DataFrame,
    players: pd.DataFrame,
    target_seasons: list[int],
    config: TrainConfig,
) -> list[PreparedSample]:
    """Use only pre-season history to predict next-season football outcomes."""
    p = prepare_players(players).drop_duplicates("gsis_id", keep="last")
    p = p.set_index("gsis_id", drop=False)
    stats = prepare_weekly_stats(stats)
    season_outcomes = aggregate_player_season_outcomes(stats)

    samples: list[PreparedSample] = []

    for target_season in target_seasons:
        target = season_outcomes[
            season_outcomes["season"].eq(target_season)
            & (season_outcomes["games"] >= config.min_target_games)
        ]
        if target.empty:
            continue

        history = stats[stats["season"] < target_season]

        for row in target.itertuples(index=False):
            player_id = _normalize_player_id(row.player_id)
            position = str(row.position).upper()
            if position not in SKILL_POSITIONS:
                continue

            hist = history[history["player_id"].eq(player_id)]
            if len(hist) < config.min_history_games:
                continue
            if player_id not in p.index:
                continue

            hist = hist.tail(config.sequence_length)
            target_vector = np.array(
                [float(getattr(row, target_name)) for target_name in OUTCOME_TARGETS],
                dtype=np.float32,
            )

            samples.append(
                PreparedSample(
                    player_id=player_id,
                    player_name=str(row.player_name),
                    position=position,
                    target_season=int(target_season),
                    sequence=hist[WEEKLY_FEATURES].to_numpy(dtype=np.float32),
                    static=_static_vector(p.loc[player_id], target_season),
                    target_outcomes=target_vector,
                )
            )

    return samples


class SequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[PreparedSample],
        seq_scaler: StandardScaler,
        static_imputer: SimpleImputer,
        static_scaler: StandardScaler,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        sequence_length: int,
    ):
        self.samples = samples
        self.seq_scaler = seq_scaler
        self.static_imputer = static_imputer
        self.static_scaler = static_scaler
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        seq = self.seq_scaler.transform(sample.sequence).astype(np.float32)

        padded = np.zeros(
            (self.sequence_length, len(WEEKLY_FEATURES)),
            dtype=np.float32,
        )
        padding_mask = np.ones(self.sequence_length, dtype=bool)
        n = min(len(seq), self.sequence_length)
        padded[-n:] = seq[-n:]
        padding_mask[-n:] = False

        static = sample.static.reshape(1, -1)
        static = self.static_imputer.transform(static)
        static = self.static_scaler.transform(static)[0].astype(np.float32)

        target_scaled = (
            (sample.target_outcomes.astype(np.float32) - self.target_mean)
            / self.target_std
        )
        target_mask = position_outcome_mask(sample.position)

        return {
            "sequence": torch.from_numpy(padded),
            "padding_mask": torch.from_numpy(padding_mask),
            "position": torch.tensor(POSITION_TO_ID[sample.position], dtype=torch.long),
            "static": torch.from_numpy(static),
            "target": torch.from_numpy(target_scaled),
            "target_real": torch.from_numpy(sample.target_outcomes.astype(np.float32)),
            "target_mask": torch.from_numpy(target_mask),
        }


class FantasyTransformer(nn.Module):
    """
    Transformer encoder that returns football outcome distributions.

    Each outcome head is represented by a mean and log standard deviation in a
    standardized outcome space. The model does not know the fantasy scoring.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(len(WEEKLY_FEATURES), config.d_model)
        self.position_embedding = nn.Embedding(len(POSITION_TO_ID), config.d_model)
        self.time_embedding = nn.Embedding(config.sequence_length, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        self.static_network = nn.Sequential(
            nn.Linear(len(STATIC_FEATURES), 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )

        hidden = config.d_model + 32
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden, 192),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(192, len(OUTCOME_TARGETS) * 2),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        position: torch.Tensor,
        static: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len, _ = sequence.shape

        x = self.input_projection(sequence)
        x = x + self.position_embedding(position).unsqueeze(1)

        sequence_positions = torch.arange(seq_len, device=sequence.device)
        x = x + self.time_embedding(sequence_positions).unsqueeze(0)

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = x[:, -1, :]
        static_vec = self.static_network(static)

        out = self.outcome_head(torch.cat([pooled, static_vec], dim=-1))
        out = out.view(-1, len(OUTCOME_TARGETS), 2)

        mean = out[:, :, 0]
        # Prevent pathological uncertainty while still allowing broad outcomes.
        log_std = out[:, :, 1].clamp(min=-3.0, max=1.0)
        return mean, log_std


def masked_gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    variance = torch.exp(2.0 * log_std)
    per_target = 0.5 * (
        (target - mean) ** 2 / variance
        + 2.0 * log_std
    )
    weighted = per_target * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def _fit_scalers(train_samples: list[PreparedSample]):
    seq_values = np.concatenate([sample.sequence for sample in train_samples], axis=0)
    seq_scaler = StandardScaler().fit(seq_values)

    static_values = np.vstack([sample.static for sample in train_samples])
    static_imputer = SimpleImputer(strategy="median").fit(static_values)
    static_scaler = StandardScaler().fit(static_imputer.transform(static_values))

    target_matrix = np.vstack([sample.target_outcomes for sample in train_samples]).astype(np.float64)
    target_masks = np.vstack([position_outcome_mask(sample.position) for sample in train_samples])

    target_mean = np.zeros(len(OUTCOME_TARGETS), dtype=np.float32)
    target_std = np.ones(len(OUTCOME_TARGETS), dtype=np.float32)

    for idx in range(len(OUTCOME_TARGETS)):
        valid = target_masks[:, idx] > 0
        values = target_matrix[valid, idx]
        if len(values) == 0:
            continue
        target_mean[idx] = float(np.mean(values))
        std = float(np.std(values))
        target_std[idx] = max(std, 0.25)

    return (
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
    )


def _real_outcome_parameters(
    mean_scaled: np.ndarray,
    log_std_scaled: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    positions: list[str],
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    means = mean_scaled * target_std + target_mean
    stds = np.exp(log_std_scaled) * target_std

    means = np.maximum(means, 0.0)
    stds = np.maximum(stds, 0.01)

    for row_idx, position in enumerate(positions):
        mask = position_outcome_mask(position).astype(bool)
        means[row_idx, ~mask] = 0.0
        stds[row_idx, ~mask] = 0.0

    games_idx = OUTCOME_TO_INDEX["games"]
    means[:, games_idx] = np.clip(
        means[:, games_idx],
        0.0,
        float(config.max_regular_season_games),
    )
    stds[:, games_idx] = np.clip(
        stds[:, games_idx],
        0.05,
        float(config.max_regular_season_games) / 2.0,
    )

    return means, stds


def fantasy_distribution_from_outcomes(
    means: np.ndarray,
    stds: np.ndarray,
    scoring: FantasyScoring,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert independent football outcome marginals into fantasy distributions.

    With linear scoring and an independence approximation, the fantasy-point
    distribution is normal with:
        mean = sum(weight_i * mean_i)
        variance = sum(weight_i^2 * variance_i)

    Correlated Monte Carlo sampling is a later upgrade.
    """
    coefficients = fantasy_scoring_coefficients(scoring)
    fantasy_mean = means @ coefficients
    fantasy_variance = (stds ** 2) @ (coefficients ** 2)
    fantasy_std = np.sqrt(np.maximum(fantasy_variance, 0.0))

    # 10th / 90th percentiles for a normal approximation.
    z80 = 1.2815515655446004
    floor = np.maximum(0.0, fantasy_mean - z80 * fantasy_std)
    ceiling = np.maximum(0.0, fantasy_mean + z80 * fantasy_std)
    return fantasy_mean, fantasy_std, floor, ceiling


def add_fantasy_projection(
    outcomes: pd.DataFrame,
    scoring: FantasyScoring,
) -> pd.DataFrame:
    """Apply league scoring to a DataFrame of Player Outcome Distributions."""
    if outcomes.empty:
        return outcomes.copy()

    df = outcomes.copy()
    means = np.column_stack(
        [pd.to_numeric(df[f"mean_{name}"], errors="coerce").fillna(0.0) for name in OUTCOME_TARGETS]
    )
    stds = np.column_stack(
        [pd.to_numeric(df[f"std_{name}"], errors="coerce").fillna(0.0) for name in OUTCOME_TARGETS]
    )

    fantasy_mean, fantasy_std, floor, ceiling = fantasy_distribution_from_outcomes(
        means,
        stds,
        scoring,
    )

    df["projected_points"] = np.maximum(fantasy_mean, 0.0)
    df["uncertainty"] = fantasy_std
    df["floor"] = floor
    df["ceiling"] = ceiling
    return df


@torch.no_grad()
def evaluate_model(
    model: FantasyTransformer,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    scoring: FantasyScoring,
    config: TrainConfig,
) -> dict:
    model.eval()

    scaled_abs_errors: list[np.ndarray] = []
    real_actual: list[np.ndarray] = []
    real_pred: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    fantasy_actual: list[float] = []
    fantasy_pred: list[float] = []

    for batch in loader:
        sequence = batch["sequence"].to(device)
        position = batch["position"].to(device)
        static = batch["static"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        target_scaled = batch["target"].to(device)
        target_real = batch["target_real"].cpu().numpy()
        target_mask = batch["target_mask"].to(device)

        mean_scaled, log_std_scaled = model(
            sequence,
            position,
            static,
            padding_mask,
        )

        scaled_error = (
            torch.abs(mean_scaled - target_scaled) * target_mask
        ).cpu().numpy()
        scaled_abs_errors.append(scaled_error)

        positions = [
            SKILL_POSITIONS[int(idx)]
            for idx in position.cpu().numpy().tolist()
        ]
        pred_mean, _ = _real_outcome_parameters(
            mean_scaled.cpu().numpy(),
            log_std_scaled.cpu().numpy(),
            target_mean,
            target_std,
            positions,
            config,
        )

        mask_np = target_mask.cpu().numpy()
        real_actual.append(target_real)
        real_pred.append(pred_mean)
        masks.append(mask_np)

        fantasy_actual.extend(score_outcome_vector(target_real, scoring).tolist())
        fantasy_pred.extend(score_outcome_vector(pred_mean, scoring).tolist())

    actual_matrix = np.vstack(real_actual)
    pred_matrix = np.vstack(real_pred)
    mask_matrix = np.vstack(masks)
    scaled_matrix = np.vstack(scaled_abs_errors)

    valid_count = np.maximum(mask_matrix.sum(), 1.0)
    normalized_mae = float(scaled_matrix.sum() / valid_count)

    outcome_mae: dict[str, float] = {}
    for idx, name in enumerate(OUTCOME_TARGETS):
        valid = mask_matrix[:, idx] > 0
        if valid.any():
            outcome_mae[name] = float(
                mean_absolute_error(actual_matrix[valid, idx], pred_matrix[valid, idx])
            )

    metrics = {
        # Kept as mae/rmse so the current Streamlit page remains compatible.
        "mae": float(mean_absolute_error(fantasy_actual, fantasy_pred)),
        "rmse": float(math.sqrt(mean_squared_error(fantasy_actual, fantasy_pred))),
        "mean_normalized_outcome_mae": normalized_mae,
        "outcome_mae": outcome_mae,
    }
    return metrics


def train_transformer(
    samples: list[PreparedSample],
    config: TrainConfig,
    scoring: FantasyScoring,
    progress: Optional[Callable[[str], None]] = None,
):
    if not samples:
        raise ValueError("No veteran training samples were created.")

    seasons = sorted({sample.target_season for sample in samples})
    if len(seasons) < 2:
        raise ValueError("Need at least two target seasons for train/validation split.")

    validation_season = seasons[-1]
    train_samples = [sample for sample in samples if sample.target_season < validation_season]
    val_samples = [sample for sample in samples if sample.target_season == validation_season]

    (
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
    ) = _fit_scalers(train_samples)

    train_ds = SequenceDataset(
        train_samples,
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
        config.sequence_length,
    )
    val_ds = SequenceDataset(
        val_samples,
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
        config.sequence_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = choose_device()
    model = FantasyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_outcome_mae = float("inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            sequence = batch["sequence"].to(device)
            position = batch["position"].to(device)
            static = batch["static"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            mean, log_std = model(sequence, position, static, padding_mask)
            loss = masked_gaussian_nll(mean, log_std, target, target_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            target_mean,
            target_std,
            scoring,
            config,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_outcome_mae": val_metrics["mean_normalized_outcome_mae"],
                "val_fantasy_mae": val_metrics["mae"],
                "val_fantasy_rmse": val_metrics["rmse"],
            }
        )

        if progress:
            progress(
                f"Epoch {epoch}/{config.epochs} — "
                f"normalized outcome MAE: {val_metrics['mean_normalized_outcome_mae']:.3f} · "
                f"fantasy MAE: {val_metrics['mae']:.1f}"
            )

        selection_metric = val_metrics["mean_normalized_outcome_mae"]
        if selection_metric < best_outcome_mae:
            best_outcome_mae = selection_metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_model(
        model,
        val_loader,
        device,
        target_mean,
        target_std,
        scoring,
        config,
    )
    final_metrics.update(
        {
            "validation_season": validation_season,
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "device": str(device),
        }
    )

    return (
        model,
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
        final_metrics,
        pd.DataFrame(history),
    )


def _build_inference_sample(
    player_id: str,
    player_name: str,
    position: str,
    target_season: int,
    stats: pd.DataFrame,
    player_row: pd.Series,
    config: TrainConfig,
) -> Optional[PreparedSample]:
    history = stats[
        stats["player_id"].eq(str(player_id))
        & (stats["season"] < target_season)
    ].tail(config.sequence_length)

    if len(history) < config.min_history_games:
        return None

    return PreparedSample(
        player_id=str(player_id),
        player_name=str(player_name),
        position=str(position),
        target_season=target_season,
        sequence=history[WEEKLY_FEATURES].to_numpy(dtype=np.float32),
        static=_static_vector(player_row, target_season),
        target_outcomes=np.zeros(len(OUTCOME_TARGETS), dtype=np.float32),
    )


@torch.no_grad()
def project_veteran_outcomes(
    model: FantasyTransformer,
    stats: pd.DataFrame,
    players: pd.DataFrame,
    target_season: int,
    config: TrainConfig,
    seq_scaler: StandardScaler,
    static_imputer: SimpleImputer,
    static_scaler: StandardScaler,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> pd.DataFrame:
    stats = prepare_weekly_stats(stats)
    players = prepare_players(players)

    diagnostics = {
        "veteran_source_season": None,
        "veteran_candidate_count": 0,
        "veteran_samples_built": 0,
        "veteran_metadata_matches": 0,
        "veteran_metadata_misses": 0,
    }

    prior_seasons = sorted(
        int(value)
        for value in stats["season"].dropna().unique()
        if int(value) < int(target_season)
    )
    if not prior_seasons:
        empty = pd.DataFrame()
        empty.attrs.update(diagnostics)
        return empty

    source_season = prior_seasons[-1]
    diagnostics["veteran_source_season"] = source_season
    previous = stats[stats["season"].eq(source_season)].copy()
    name_col = _player_name_column(previous)

    candidates = (
        previous[["player_id", name_col, "position"]]
        .rename(columns={name_col: "player_name"})
        .drop_duplicates("player_id", keep="last")
        .copy()
    )
    candidates["player_id"] = candidates["player_id"].map(_normalize_player_id)
    candidates = candidates[candidates["player_id"].ne("")]
    diagnostics["veteran_candidate_count"] = int(len(candidates))

    players = players.drop_duplicates("gsis_id", keep="last")
    player_lookup = players.set_index("gsis_id", drop=False)

    samples: list[PreparedSample] = []
    for row in candidates.itertuples(index=False):
        player_id = _normalize_player_id(row.player_id)
        position = str(row.position).upper()
        if position not in SKILL_POSITIONS:
            continue

        if player_id in player_lookup.index:
            player_row = player_lookup.loc[player_id]
            diagnostics["veteran_metadata_matches"] += 1
        else:
            player_row = pd.Series(
                {
                    "gsis_id": player_id,
                    "display_name": row.player_name,
                    "position": position,
                    "birth_date": pd.NaT,
                    "rookie_season": np.nan,
                    "draft_year": np.nan,
                    "height": np.nan,
                    "weight": np.nan,
                    "draft_round": np.nan,
                    "draft_pick": np.nan,
                }
            )
            diagnostics["veteran_metadata_misses"] += 1

        sample = _build_inference_sample(
            player_id,
            row.player_name,
            position,
            target_season,
            stats,
            player_row,
            config,
        )
        if sample is not None:
            samples.append(sample)

    diagnostics["veteran_samples_built"] = int(len(samples))
    if not samples:
        empty = pd.DataFrame()
        empty.attrs.update(diagnostics)
        return empty

    dataset = SequenceDataset(
        samples,
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
        config.sequence_length,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    device = next(model.parameters()).device
    all_means = []
    all_stds = []
    cursor = 0

    model.eval()
    for batch in loader:
        size = batch["position"].shape[0]
        positions = [sample.position for sample in samples[cursor:cursor + size]]
        cursor += size

        mean_scaled, log_std_scaled = model(
            batch["sequence"].to(device),
            batch["position"].to(device),
            batch["static"].to(device),
            batch["padding_mask"].to(device),
        )

        means, stds = _real_outcome_parameters(
            mean_scaled.cpu().numpy(),
            log_std_scaled.cpu().numpy(),
            target_mean,
            target_std,
            positions,
            config,
        )
        all_means.append(means)
        all_stds.append(stds)

    means = np.vstack(all_means)
    stds = np.vstack(all_stds)

    rows = []
    for row_idx, sample in enumerate(samples):
        row = {
            "player_id": sample.player_id,
            "player": sample.player_name,
            "position": sample.position,
            "rookie": False,
            "model": "transformer_outcomes",
        }
        for target_idx, target_name in enumerate(OUTCOME_TARGETS):
            row[f"mean_{target_name}"] = float(means[row_idx, target_idx])
            row[f"std_{target_name}"] = float(stds[row_idx, target_idx])
        rows.append(row)

    result = pd.DataFrame(rows)

    if "sleeper_id" in players.columns and not result.empty:
        id_map = (
            players[["gsis_id", "sleeper_id"]]
            .copy()
            .drop_duplicates("gsis_id", keep="last")
        )
        id_map["sleeper_id"] = id_map["sleeper_id"].map(_normalize_sleeper_id)
        result = result.merge(
            id_map,
            left_on="player_id",
            right_on="gsis_id",
            how="left",
        ).drop(columns=["gsis_id"], errors="ignore")
        result["sleeper_id"] = result["sleeper_id"].fillna("").map(_normalize_sleeper_id)
    elif "sleeper_id" not in result.columns:
        result["sleeper_id"] = ""

    result.attrs.update(diagnostics)
    return result


def _height_to_inches(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if 55 <= number <= 90:
            return number
    text = str(value).strip()
    match = re.search(r"(\d+)\D+(\d+)", text)
    if match:
        return float(match.group(1)) * 12.0 + float(match.group(2))
    return pd.to_numeric(text, errors="coerce")


def prepare_rookie_table(
    players: pd.DataFrame,
    combine: pd.DataFrame,
    draft: pd.DataFrame,
) -> pd.DataFrame:
    """Build rookie draft/combine features with GSIS IDs where available."""
    p = prepare_players(players).copy()
    c = combine.copy()
    d = draft.copy()

    if "season" not in d.columns:
        d["season"] = np.nan
    if "round" not in d.columns:
        d["round"] = np.nan
    if "pick" not in d.columns:
        d["pick"] = np.nan
    if "position" not in d.columns:
        d["position"] = ""
    if "pfr_player_name" not in d.columns:
        d["pfr_player_name"] = ""
    if "pfr_player_id" not in d.columns:
        d["pfr_player_id"] = np.nan

    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["round"] = pd.to_numeric(d["round"], errors="coerce")
    d["pick"] = pd.to_numeric(d["pick"], errors="coerce")
    d["position"] = d["position"].astype(str).str.upper()
    d["name_key"] = d["pfr_player_name"].map(_normalize_name)

    if "season" not in c.columns:
        c["season"] = np.nan
    if "pos" not in c.columns:
        c["pos"] = ""
    if "player_name" not in c.columns:
        c["player_name"] = ""
    if "pfr_id" not in c.columns:
        c["pfr_id"] = np.nan
    if "ht" not in c.columns:
        c["ht"] = np.nan

    c["season"] = pd.to_numeric(c["season"], errors="coerce")
    c["pos"] = c["pos"].astype(str).str.upper()
    c["name_key"] = c["player_name"].map(_normalize_name)
    c["height_combine"] = c["ht"].map(_height_to_inches)

    for col in [
        "wt",
        "forty",
        "bench",
        "vertical",
        "broad_jump",
        "cone",
        "shuttle",
        "draft_round",
        "draft_ovr",
    ]:
        if col not in c.columns:
            c[col] = np.nan
        c[col] = pd.to_numeric(c[col], errors="coerce")

    p["draft_year"] = pd.to_numeric(p["draft_year"], errors="coerce")
    p["draft_round"] = pd.to_numeric(p["draft_round"], errors="coerce")
    p["draft_pick"] = pd.to_numeric(p["draft_pick"], errors="coerce")
    p["name_key"] = p["display_name"].map(_normalize_name)

    rookies = p[
        p["position"].isin(SKILL_POSITIONS)
        & p["draft_year"].notna()
    ].copy()

    c_by_id = c[c["pfr_id"].notna()].copy()
    if "pfr_id" in rookies.columns and not c_by_id.empty:
        cols = [
            "pfr_id",
            "season",
            "height_combine",
            "wt",
            "forty",
            "bench",
            "vertical",
            "broad_jump",
            "cone",
            "shuttle",
        ]
        c_by_id = c_by_id[cols].drop_duplicates("pfr_id", keep="last")
        rookies = rookies.merge(c_by_id, on="pfr_id", how="left", suffixes=("", "_combine"))
    else:
        for col in [
            "height_combine",
            "wt",
            "forty",
            "bench",
            "vertical",
            "broad_jump",
            "cone",
            "shuttle",
        ]:
            rookies[col] = np.nan

    combine_name = c[
        [
            "name_key",
            "season",
            "height_combine",
            "wt",
            "forty",
            "bench",
            "vertical",
            "broad_jump",
            "cone",
            "shuttle",
        ]
    ].drop_duplicates(["name_key", "season"], keep="last")

    rookies = rookies.merge(
        combine_name,
        left_on=["name_key", "draft_year"],
        right_on=["name_key", "season"],
        how="left",
        suffixes=("", "_name"),
    )

    for col in [
        "height_combine",
        "wt",
        "forty",
        "bench",
        "vertical",
        "broad_jump",
        "cone",
        "shuttle",
    ]:
        alt = f"{col}_name"
        if alt in rookies.columns:
            rookies[col] = rookies[col].combine_first(rookies[alt])

    rookies["height"] = rookies["height"].combine_first(rookies["height_combine"])
    rookies["weight"] = rookies["weight"].combine_first(rookies["wt"])

    if "pfr_id" in rookies.columns:
        d_by_id = d[d["pfr_player_id"].notna()].drop_duplicates("pfr_player_id", keep="last")
        rookies = rookies.merge(
            d_by_id[["pfr_player_id", "round", "pick"]],
            left_on="pfr_id",
            right_on="pfr_player_id",
            how="left",
            suffixes=("", "_drafttable"),
        )
        rookies["draft_round"] = rookies["draft_round"].combine_first(rookies["round"])
        rookies["draft_pick"] = rookies["draft_pick"].combine_first(rookies["pick"])

    for col in ROOKIE_NUMERIC_FEATURES:
        if col not in rookies.columns:
            rookies[col] = np.nan
        rookies[col] = pd.to_numeric(rookies[col], errors="coerce")

    rookies["draft_round"] = rookies["draft_round"].fillna(8.0)
    rookies["draft_pick"] = rookies["draft_pick"].fillna(300.0)
    rookies["position_id"] = rookies["position"].map(POSITION_TO_ID).astype(float)
    return rookies


def train_rookie_outcome_models(
    rookie_table: pd.DataFrame,
    stats: pd.DataFrame,
    target_season: int,
    config: TrainConfig,
):
    """Train one small RF per outcome using historical rookie seasons."""
    season_outcomes = aggregate_player_season_outcomes(stats)

    historical = rookie_table[rookie_table["draft_year"] < target_season].copy()
    historical["draft_year"] = pd.to_numeric(historical["draft_year"], errors="coerce")

    historical = historical.merge(
        season_outcomes,
        left_on=["gsis_id", "draft_year"],
        right_on=["player_id", "season"],
        how="inner",
        suffixes=("", "_target"),
    )
    historical = historical[historical["games"] >= 1].copy()

    features = ["position_id"] + ROOKIE_NUMERIC_FEATURES
    models: dict[str, Pipeline] = {}
    sample_counts: dict[str, int] = {}

    for target_name in OUTCOME_TARGETS:
        relevant_positions = {
            position
            for position, targets in POSITION_OUTCOMES.items()
            if target_name in targets
        }
        subset = historical[historical["position"].isin(relevant_positions)].copy()
        if len(subset) < 12:
            continue

        X = subset[features]
        y = pd.to_numeric(subset[target_name], errors="coerce").fillna(0.0)

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=config.rookie_trees,
                        min_samples_leaf=3,
                        max_features=0.8,
                        random_state=config.random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)
        models[target_name] = pipeline
        sample_counts[target_name] = len(subset)

    return models, sample_counts, int(len(historical))


def project_rookie_outcomes(
    rookie_models: dict[str, Pipeline],
    rookie_table: pd.DataFrame,
    target_season: int,
    config: TrainConfig,
) -> pd.DataFrame:
    current = rookie_table[
        rookie_table["draft_year"].eq(target_season)
        & rookie_table["position"].isin(SKILL_POSITIONS)
    ].copy()
    if current.empty:
        return pd.DataFrame()

    features = ["position_id"] + ROOKIE_NUMERIC_FEATURES
    X = current[features]

    rows = []
    for row_idx, player in current.reset_index(drop=True).iterrows():
        row = {
            "player_id": str(player["gsis_id"]),
            "sleeper_id": _normalize_sleeper_id(player.get("sleeper_id", "")),
            "player": str(player["display_name"]),
            "position": str(player["position"]),
            "rookie": True,
            "model": "rookie_outcome_prior",
            "draft_pick": player.get("draft_pick", np.nan),
            "draft_round": player.get("draft_round", np.nan),
        }
        for target_name in OUTCOME_TARGETS:
            row[f"mean_{target_name}"] = 0.0
            row[f"std_{target_name}"] = 0.0
        rows.append(row)

    for target_name, pipeline in rookie_models.items():
        pred = np.maximum(pipeline.predict(X), 0.0)

        imputer = pipeline.named_steps["imputer"]
        forest = pipeline.named_steps["model"]
        transformed = imputer.transform(X)
        tree_predictions = np.vstack([
            tree.predict(transformed) for tree in forest.estimators_
        ])
        std = np.maximum(tree_predictions.std(axis=0), 0.01)

        for idx, player in current.reset_index(drop=True).iterrows():
            position = str(player["position"])
            if target_name not in POSITION_OUTCOMES[position]:
                continue

            mean_value = float(pred[idx])
            std_value = float(std[idx])
            if target_name == "games":
                mean_value = float(np.clip(mean_value, 0.0, config.max_regular_season_games))
                std_value = float(np.clip(std_value, 0.05, config.max_regular_season_games / 2.0))

            rows[idx][f"mean_{target_name}"] = mean_value
            rows[idx][f"std_{target_name}"] = std_value

    return pd.DataFrame(rows)


def add_draft_value(rankings: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    if rankings.empty:
        return rankings

    df = rankings.copy()
    replacement_ranks = {
        "QB": config.replacement_rank_qb,
        "RB": config.replacement_rank_rb,
        "WR": config.replacement_rank_wr,
        "TE": config.replacement_rank_te,
    }

    replacement = {}
    for position, rank in replacement_ranks.items():
        values = (
            df[df["position"].eq(position)]["projected_points"]
            .sort_values(ascending=False)
            .to_numpy()
        )
        replacement[position] = (
            float(values[min(rank - 1, len(values) - 1)])
            if len(values)
            else 0.0
        )

    df["replacement_points"] = df["position"].map(replacement).fillna(0.0)
    df["vorp"] = df["projected_points"] - df["replacement_points"]

    df = df.sort_values(
        ["vorp", "projected_points"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df.insert(0, "overall_rank", np.arange(1, len(df) + 1))
    df["position_rank"] = df.groupby("position").cumcount() + 1
    df["pos_rank"] = df["position"] + df["position_rank"].astype(str)
    return df


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_artifacts(
    output_dir: Path,
    target_season: int,
    scoring: FantasyScoring,
    config: TrainConfig,
    model: FantasyTransformer,
    seq_scaler,
    static_imputer,
    static_scaler,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    rookie_models,
    metrics: dict,
    outcomes: pd.DataFrame,
    rankings: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{target_season}_ppr{scoring.reception:g}"

    torch_path = output_dir / f"player_outcome_transformer_{target_season}.pt"
    sklearn_path = output_dir / f"outcome_preprocessing_and_rookies_{target_season}.joblib"
    outcomes_path = output_dir / f"player_outcomes_{target_season}.csv"
    rankings_path = output_dir / f"draft_rankings_{tag}.csv"
    metadata_path = output_dir / f"outcome_metadata_{tag}.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "weekly_features": WEEKLY_FEATURES,
            "static_features": STATIC_FEATURES,
            "outcome_targets": OUTCOME_TARGETS,
            "target_mean": np.asarray(target_mean),
            "target_std": np.asarray(target_std),
            "architecture": "player_outcome_distributions_v1",
        },
        torch_path,
    )

    joblib.dump(
        {
            "seq_scaler": seq_scaler,
            "static_imputer": static_imputer,
            "static_scaler": static_scaler,
            "rookie_outcome_models": rookie_models,
        },
        sklearn_path,
    )

    outcomes.to_csv(outcomes_path, index=False)
    rankings.to_csv(rankings_path, index=False)

    metadata = {
        "architecture": "player_outcome_distributions_v1",
        "target_season": target_season,
        "scoring_used_for_rankings": asdict(scoring),
        "config": asdict(config),
        "outcome_targets": OUTCOME_TARGETS,
        "metrics": _json_safe(metrics),
        "notes": (
            "Transformer training is scoring-agnostic. Sleeper scoring is applied "
            "downstream to outcome distributions. Outcome correlations are not yet modeled."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "transformer": str(torch_path),
        "preprocessing": str(sklearn_path),
        "outcomes": str(outcomes_path),
        "rankings": str(rankings_path),
        "metadata": str(metadata_path),
    }


def train_and_rank(
    target_season: int,
    config: Optional[TrainConfig] = None,
    scoring: Optional[FantasyScoring] = None,
    output_dir: str | Path = "models",
    source: str = "nflverse",
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Train a scoring-independent Player Outcome Transformer, then apply the
    selected league scoring to build draft rankings.
    """
    config = config or TrainConfig()
    scoring = scoring or FantasyScoring()
    output_dir = Path(output_dir)
    set_seed(config.random_seed)

    stats, players, combine, draft = load_sportsdataverse(
        start_season=config.start_season,
        end_season=target_season,
        source=source,
        progress=progress,
    )
    stats = prepare_weekly_stats(stats)

    historical_target_seasons = sorted(
        int(value)
        for value in stats["season"].dropna().unique()
        if config.start_season + 1 <= int(value) < target_season
    )
    if len(historical_target_seasons) < 3:
        raise ValueError(
            "Not enough historical seasons. Choose an earlier start season or a later target season."
        )

    if progress:
        progress("Building leakage-safe veteran player-season outcome samples...")
    samples = build_veteran_training_samples(
        stats,
        players,
        historical_target_seasons,
        config,
    )

    if progress:
        progress(
            f"Training Player Outcome Transformer on {len(samples):,} veteran player-season samples..."
        )

    (
        model,
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
        metrics,
        history,
    ) = train_transformer(samples, config, scoring, progress=progress)

    if progress:
        progress(f"Projecting veteran football outcomes for {target_season}...")
    veteran_outcomes = project_veteran_outcomes(
        model=model,
        stats=stats,
        players=players,
        target_season=target_season,
        config=config,
        seq_scaler=seq_scaler,
        static_imputer=static_imputer,
        static_scaler=static_scaler,
        target_mean=target_mean,
        target_std=target_std,
    )
    veteran_diagnostics = dict(veteran_outcomes.attrs)

    if progress:
        progress("Training rookie football-outcome priors from historical draft/combine classes...")
    rookie_table = prepare_rookie_table(players, combine, draft)
    rookie_models, rookie_target_samples, rookie_training_samples = train_rookie_outcome_models(
        rookie_table,
        stats,
        target_season,
        config,
    )
    rookie_outcomes = project_rookie_outcomes(
        rookie_models,
        rookie_table,
        target_season,
        config,
    )

    outcomes = pd.concat(
        [veteran_outcomes, rookie_outcomes],
        ignore_index=True,
        sort=False,
    )

    if progress:
        progress("Applying Sleeper fantasy scoring to the football outcome distributions...")
    scored = add_fantasy_projection(outcomes, scoring)
    rankings = add_draft_value(scored, config)

    veteran_ids = set(veteran_outcomes.get("player_id", pd.Series(dtype=str)).astype(str))
    rookie_ids = set(rookie_outcomes.get("player_id", pd.Series(dtype=str)).astype(str))
    veteran_rankings = rankings[rankings["player_id"].astype(str).isin(veteran_ids)].copy()
    rookie_rankings = rankings[rankings["player_id"].astype(str).isin(rookie_ids)].copy()

    metrics.update(veteran_diagnostics)
    metrics["rookie_training_samples"] = rookie_training_samples
    metrics["rookie_outcome_sample_counts"] = rookie_target_samples
    metrics["ranked_veterans"] = int((~rankings["rookie"]).sum()) if not rankings.empty else 0
    metrics["ranked_rookies"] = int(rankings["rookie"].sum()) if not rankings.empty else 0
    metrics["architecture"] = "player_outcome_distributions_v1"

    paths = save_artifacts(
        output_dir=output_dir,
        target_season=target_season,
        scoring=scoring,
        config=config,
        model=model,
        seq_scaler=seq_scaler,
        static_imputer=static_imputer,
        static_scaler=static_scaler,
        target_mean=target_mean,
        target_std=target_std,
        rookie_models=rookie_models,
        metrics=metrics,
        outcomes=outcomes,
        rankings=rankings,
    )

    if progress:
        progress("Training complete. Player outcome distributions and draft rankings are ready.")

    return {
        "rankings": rankings,
        "outcomes": outcomes,
        "veteran_outcomes": veteran_outcomes,
        "rookie_outcomes": rookie_outcomes,
        "veteran_rankings": veteran_rankings,
        "rookie_rankings": rookie_rankings,
        "metrics": metrics,
        "history": history,
        "paths": paths,
    }
