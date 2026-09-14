"""
fantasy_transformer.py

Draft-oriented NFL fantasy model.

Design:
- SportsDataverse supplies weekly NFL player stats and player metadata.
- Veteran players are modeled with a Transformer over their recent weekly history.
- The Transformer predicts NEXT-SEASON fantasy points, not a direct rank.
- Rookie players are handled by a separate rookie model using draft capital,
  combine measurements, size, and position because they have no NFL history.
- Veteran + rookie projections are merged and converted to VORP-style draft ranks.

This is intentionally a strong MVP rather than a final production model.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
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

# Weekly football features. Missing columns are safely created as zeros so the
# model survives modest upstream schema differences.
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
    "fantasy_points_model",
]

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
    """Basic scoring settings. Later these can be filled from Sleeper."""
    reception: float = 1.0
    passing_yard: float = 0.04
    passing_td: float = 4.0
    interception: float = -2.0
    rushing_yard: float = 0.10
    rushing_td: float = 6.0
    receiving_yard: float = 0.10
    receiving_td: float = 6.0
    two_point_conversion: float = 2.0
    fumble_lost: float = -2.0


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
    # Approximate 12-team, 1-QB replacement levels for VORP.
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
    target_points: float
    target_games: int


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
    """Normalize GSIS/player IDs before joining stats to the player master."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return value


def add_fantasy_points(stats: pd.DataFrame, scoring: FantasyScoring) -> pd.DataFrame:
    """Calculate fantasy points from football stats using configurable scoring."""
    stats = stats.copy()
    needed = [
        "passing_yards", "passing_tds", "interceptions",
        "rushing_yards", "rushing_tds", "receptions",
        "receiving_yards", "receiving_tds",
        "passing_2pt_conversions", "rushing_2pt_conversions",
        "receiving_2pt_conversions", "sack_fumbles_lost",
        "rushing_fumbles_lost", "receiving_fumbles_lost",
    ]
    stats = _to_numeric(stats, needed)
    x = stats.fillna({c: 0.0 for c in needed})

    # The three fumble-lost fields represent different play roles. Summing is a
    # reasonable first approximation for an offensive player-week.
    fumbles_lost = (
        x["sack_fumbles_lost"]
        + x["rushing_fumbles_lost"]
        + x["receiving_fumbles_lost"]
    )

    two_pt = (
        x["passing_2pt_conversions"]
        + x["rushing_2pt_conversions"]
        + x["receiving_2pt_conversions"]
    )

    stats["fantasy_points_model"] = (
        x["passing_yards"] * scoring.passing_yard
        + x["passing_tds"] * scoring.passing_td
        + x["interceptions"] * scoring.interception
        + x["rushing_yards"] * scoring.rushing_yard
        + x["rushing_tds"] * scoring.rushing_td
        + x["receptions"] * scoring.reception
        + x["receiving_yards"] * scoring.receiving_yard
        + x["receiving_tds"] * scoring.receiving_td
        + two_pt * scoring.two_point_conversion
        + fumbles_lost * scoring.fumble_lost
    )
    return stats


def load_sportsdataverse(
    start_season: int,
    end_season: int,
    scoring: FantasyScoring,
    source: str = "nflverse",
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data needed by the MVP.

    SportsDataverse's current NFL loaders mirror nflreadpy/nflverse. Player
    stats are delivered as one combined week-level dataset, so we load once and
    filter locally.
    """
    if progress:
        progress("Loading SportsDataverse weekly NFL player stats...")

    from sportsdataverse.nfl import (
        load_nfl_combine,
        load_nfl_draft_picks,
        load_nfl_player_stats,
        load_nfl_players,
    )

    stats = load_nfl_player_stats(
        return_as_pandas=True,
        source=source,
    )
    stats = stats[
        (pd.to_numeric(stats["season"], errors="coerce") >= start_season)
        & (pd.to_numeric(stats["season"], errors="coerce") <= end_season)
    ].copy()

    if progress:
        progress("Loading SportsDataverse player identity data...")
    players = load_nfl_players(return_as_pandas=True, source=source)

    if progress:
        progress("Loading NFL combine and draft-capital data for rookies...")
    combine = load_nfl_combine(return_as_pandas=True)
    draft = load_nfl_draft_picks(return_as_pandas=True)

    stats = add_fantasy_points(stats, scoring)
    return stats, players, combine, draft


def prepare_weekly_stats(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()

    if "player_id" not in df.columns:
        raise ValueError("SportsDataverse player stats are missing player_id.")

    df["player_id"] = df["player_id"].map(_normalize_player_id)
    df = df[df["player_id"].ne("")].copy()

    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().eq("REG")]

    df["position"] = df["position"].astype(str).str.upper()
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()

    numeric = list(set(WEEKLY_FEATURES + ["season", "week", "fantasy_points_model"]))
    df = _to_numeric(df, numeric)

    # Ratios occasionally contain inf from source calculations.
    df[WEEKLY_FEATURES] = (
        df[WEEKLY_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Stable sort establishes temporal order across seasons.
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return df


def prepare_players(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    if "gsis_id" not in p.columns:
        raise ValueError("SportsDataverse player master is missing gsis_id.")

    p["gsis_id"] = p["gsis_id"].map(_normalize_player_id)
    p = p[p["gsis_id"].ne("")].copy()
    p["position"] = p["position"].astype(str).str.upper()

    for col in ["height", "weight", "rookie_season", "draft_year", "draft_round", "draft_pick"]:
        if col not in p.columns:
            p[col] = np.nan
        p[col] = pd.to_numeric(p[col], errors="coerce")

    if "birth_date" not in p.columns:
        p["birth_date"] = pd.NaT
    p["birth_date"] = pd.to_datetime(p["birth_date"], errors="coerce")

    if "display_name" not in p.columns:
        p["display_name"] = ""
    return p


def _static_vector(player_row: pd.Series, target_season: int) -> np.ndarray:
    birth = player_row.get("birth_date", pd.NaT)
    if pd.isna(birth):
        age = np.nan
    else:
        age = target_season - float(birth.year) + 0.5

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

    # Undrafted defaults preserve useful ordering rather than treating missing
    # draft capital as "better than pick 1".
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


def build_veteran_training_samples(
    stats: pd.DataFrame,
    players: pd.DataFrame,
    target_seasons: list[int],
    config: TrainConfig,
) -> list[PreparedSample]:
    """Each sample uses only weeks BEFORE target_season to predict target season total."""
    p = prepare_players(players).set_index("gsis_id", drop=False)
    stats = prepare_weekly_stats(stats)

    samples: list[PreparedSample] = []

    for target_season in target_seasons:
        target = stats[stats["season"].eq(target_season)]
        if target.empty:
            continue

        target_summary = (
            target.groupby(["player_id", "player_name", "position"], dropna=False)
            .agg(
                target_points=("fantasy_points_model", "sum"),
                target_games=("week", "nunique"),
            )
            .reset_index()
        )
        target_summary = target_summary[
            target_summary["target_games"] >= config.min_target_games
        ]

        history = stats[stats["season"] < target_season]

        for row in target_summary.itertuples(index=False):
            player_id = str(row.player_id)
            hist = history[history["player_id"].astype(str).eq(player_id)]
            if len(hist) < config.min_history_games:
                continue
            if player_id not in p.index:
                continue

            hist = hist.tail(config.sequence_length)
            seq = hist[WEEKLY_FEATURES].to_numpy(dtype=np.float32)
            static = _static_vector(p.loc[player_id], target_season)

            samples.append(
                PreparedSample(
                    player_id=player_id,
                    player_name=str(row.player_name),
                    position=str(row.position),
                    target_season=int(target_season),
                    sequence=seq,
                    static=static,
                    target_points=float(row.target_points),
                    target_games=int(row.target_games),
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
        sequence_length: int,
    ):
        self.samples = samples
        self.seq_scaler = seq_scaler
        self.static_imputer = static_imputer
        self.static_scaler = static_scaler
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        seq = self.seq_scaler.transform(s.sequence).astype(np.float32)

        padded = np.zeros(
            (self.sequence_length, len(WEEKLY_FEATURES)),
            dtype=np.float32,
        )
        mask = np.ones(self.sequence_length, dtype=bool)

        n = min(len(seq), self.sequence_length)
        padded[-n:] = seq[-n:]
        mask[-n:] = False  # Transformer convention: True means padding.

        static = s.static.reshape(1, -1)
        static = self.static_imputer.transform(static)
        static = self.static_scaler.transform(static)[0].astype(np.float32)

        return {
            "sequence": torch.from_numpy(padded),
            "padding_mask": torch.from_numpy(mask),
            "position": torch.tensor(POSITION_TO_ID[s.position], dtype=torch.long),
            "static": torch.from_numpy(static),
            "target": torch.tensor(s.target_points, dtype=torch.float32),
        }


class FantasyTransformer(nn.Module):
    """
    Transformer encoder for veteran next-season fantasy production.

    It outputs (mean, log_std) so the downstream system gets both an expected
    score and a first approximation of uncertainty.
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

        self.head = nn.Sequential(
            nn.Linear(config.d_model + 32, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        position: torch.Tensor,
        static: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = sequence.shape

        x = self.input_projection(sequence)
        x = x + self.position_embedding(position).unsqueeze(1)

        positions = torch.arange(seq_len, device=sequence.device)
        x = x + self.time_embedding(positions).unsqueeze(0)

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Last slot is always a real token because we left-pad.
        pooled = x[:, -1, :]
        static_vec = self.static_network(static)
        out = self.head(torch.cat([pooled, static_vec], dim=-1))

        mean = out[:, 0]
        log_std = out[:, 1].clamp(min=-2.5, max=5.0)
        return mean, log_std


def gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    var = torch.exp(2.0 * log_std)
    return torch.mean(
        0.5 * ((target - mean) ** 2 / var + 2.0 * log_std)
    )


def _fit_scalers(train_samples: list[PreparedSample]):
    seq_values = np.concatenate([s.sequence for s in train_samples], axis=0)
    seq_scaler = StandardScaler().fit(seq_values)

    static_values = np.vstack([s.static for s in train_samples])
    static_imputer = SimpleImputer(strategy="median").fit(static_values)
    static_filled = static_imputer.transform(static_values)
    static_scaler = StandardScaler().fit(static_filled)

    return seq_scaler, static_imputer, static_scaler


@torch.no_grad()
@torch.no_grad()
def evaluate_model(
    model: FantasyTransformer,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    """Evaluate veteran projections in real fantasy-point units."""
    model.eval()
    actual, pred = [], []

    for batch in loader:
        seq = batch["sequence"].to(device)
        pos = batch["position"].to(device)
        static = batch["static"].to(device)
        mask = batch["padding_mask"].to(device)
        y = batch["target"].to(device)

        mean_scaled, _ = model(seq, pos, static, mask)
        mean_points = mean_scaled * target_std + target_mean

        actual.extend(y.cpu().numpy().tolist())
        pred.extend(mean_points.cpu().numpy().tolist())

    if not actual:
        return {"mae": float("nan"), "rmse": float("nan")}

    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
    }


def train_transformer(
    samples: list[PreparedSample],
    config: TrainConfig,
    progress: Optional[Callable[[str], None]] = None,
):
    if not samples:
        raise ValueError("No veteran training samples were created.")

    seasons = sorted({s.target_season for s in samples})
    if len(seasons) < 2:
        raise ValueError("Need at least two target seasons for train/validation split.")

    validation_season = seasons[-1]
    train_samples = [s for s in samples if s.target_season < validation_season]
    val_samples = [s for s in samples if s.target_season == validation_season]

    seq_scaler, static_imputer, static_scaler = _fit_scalers(train_samples)

    # Put season fantasy-point targets onto a stable training scale.
    target_values = np.array(
        [s.target_points for s in train_samples],
        dtype=np.float32,
    )
    target_mean = float(target_values.mean())
    target_std = float(target_values.std())
    if target_std < 1e-6:
        target_std = 1.0

    train_ds = SequenceDataset(
        train_samples,
        seq_scaler,
        static_imputer,
        static_scaler,
        config.sequence_length,
    )
    val_ds = SequenceDataset(
        val_samples,
        seq_scaler,
        static_imputer,
        static_scaler,
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
    best_val_mae = float("inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            seq = batch["sequence"].to(device)
            pos = batch["position"].to(device)
            static = batch["static"].to(device)
            mask = batch["padding_mask"].to(device)
            target_points = batch["target"].to(device)

            target_scaled = (target_points - target_mean) / target_std

            optimizer.zero_grad(set_to_none=True)
            mean_scaled, log_std_scaled = model(seq, pos, static, mask)
            loss = gaussian_nll(
                mean_scaled,
                log_std_scaled,
                target_scaled,
            )
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
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )

        if progress:
            progress(
                f"Epoch {epoch}/{config.epochs} — "
                f"validation MAE: {val_metrics['mae']:.2f} fantasy points"
            )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_model(
        model,
        val_loader,
        device,
        target_mean,
        target_std,
    )
    final_metrics["validation_season"] = validation_season
    final_metrics["train_samples"] = len(train_samples)
    final_metrics["validation_samples"] = len(val_samples)
    final_metrics["device"] = str(device)
    final_metrics["veteran_target_mean"] = target_mean
    final_metrics["veteran_target_std"] = target_std

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
    hist = stats[
        stats["player_id"].astype(str).eq(str(player_id))
        & (stats["season"] < target_season)
    ].tail(config.sequence_length)

    if len(hist) < config.min_history_games:
        return None

    return PreparedSample(
        player_id=str(player_id),
        player_name=str(player_name),
        position=str(position),
        target_season=target_season,
        sequence=hist[WEEKLY_FEATURES].to_numpy(dtype=np.float32),
        static=_static_vector(player_row, target_season),
        target_points=0.0,
        target_games=0,
    )


@torch.no_grad()
def project_veterans(
    model: FantasyTransformer,
    stats: pd.DataFrame,
    players: pd.DataFrame,
    target_season: int,
    config: TrainConfig,
    seq_scaler: StandardScaler,
    static_imputer: SimpleImputer,
    static_scaler: StandardScaler,
    target_mean: float,
    target_std: float,
) -> pd.DataFrame:
    """
    Project veteran players for target_season.

    Veteran candidates come from the newest stats season before target_season.
    Missing player-master metadata does NOT remove a veteran; the static-feature
    imputer handles missing values instead.
    """
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
        int(x)
        for x in stats["season"].dropna().unique()
        if int(x) < int(target_season)
    )

    if not prior_seasons:
        empty = pd.DataFrame()
        empty.attrs.update(diagnostics)
        return empty

    source_season = prior_seasons[-1]
    diagnostics["veteran_source_season"] = source_season

    previous = stats[stats["season"].eq(source_season)].copy()

    name_col = (
        "player_display_name"
        if "player_display_name" in previous.columns
        else "player_name"
    )

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

    samples = []

    for row in candidates.itertuples(index=False):
        pid = _normalize_player_id(row.player_id)
        position = str(row.position).upper()

        if position not in SKILL_POSITIONS:
            continue

        if pid in player_lookup.index:
            player_row = player_lookup.loc[pid]
            diagnostics["veteran_metadata_matches"] += 1
        else:
            # Sequence history is enough to project the player. Missing static
            # metadata is filled by the fitted SimpleImputer.
            player_row = pd.Series(
                {
                    "gsis_id": pid,
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
            pid,
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

    ds = SequenceDataset(
        samples,
        seq_scaler,
        static_imputer,
        static_scaler,
        config.sequence_length,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    device = next(model.parameters()).device
    means, stds = [], []

    model.eval()
    for batch in loader:
        mean_scaled, log_std_scaled = model(
            batch["sequence"].to(device),
            batch["position"].to(device),
            batch["static"].to(device),
            batch["padding_mask"].to(device),
        )

        mean_points = mean_scaled * target_std + target_mean
        std_points = torch.exp(log_std_scaled) * target_std

        means.extend(mean_points.cpu().numpy())
        stds.extend(std_points.cpu().numpy())

    rows = []
    for sample, mean, std in zip(samples, means, stds):
        rows.append(
            {
                "player_id": sample.player_id,
                "player": sample.player_name,
                "position": sample.position,
                "projected_points": max(0.0, float(mean)),
                "uncertainty": max(1.0, float(std)),
                "rookie": False,
                "model": "transformer",
            }
        )

    result = pd.DataFrame(rows)
    result.attrs.update(diagnostics)
    return result


def _height_to_inches(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        # Some datasets already use inches.
        if 55 <= v <= 90:
            return v
    s = str(value).strip()
    match = re.search(r"(\d+)\D+(\d+)", s)
    if match:
        return float(match.group(1)) * 12.0 + float(match.group(2))
    return pd.to_numeric(s, errors="coerce")


def prepare_rookie_table(
    players: pd.DataFrame,
    combine: pd.DataFrame,
    draft: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a historical rookie feature table.

    Primary identity is PFR ID where possible. A normalized-name/year fallback
    is included because combine/draft tables have occasional ID gaps.
    """
    p = prepare_players(players).copy()
    c = combine.copy()
    d = draft.copy()

    # Draft table.
    d["season"] = pd.to_numeric(d.get("season"), errors="coerce")
    d["round"] = pd.to_numeric(d.get("round"), errors="coerce")
    d["pick"] = pd.to_numeric(d.get("pick"), errors="coerce")
    d["position"] = d.get("position", "").astype(str).str.upper()
    d["name_key"] = d.get("pfr_player_name", "").map(_normalize_name)

    # Combine table.
    c["season"] = pd.to_numeric(c.get("season"), errors="coerce")
    c["pos"] = c.get("pos", "").astype(str).str.upper()
    c["name_key"] = c.get("player_name", "").map(_normalize_name)
    c["height_combine"] = c.get("ht", np.nan).map(_height_to_inches)

    for col in ["wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle", "draft_round", "draft_ovr"]:
        if col not in c.columns:
            c[col] = np.nan
        c[col] = pd.to_numeric(c[col], errors="coerce")

    # Player master gives us the canonical GSIS identifier.
    p["draft_year"] = pd.to_numeric(p["draft_year"], errors="coerce")
    p["draft_round"] = pd.to_numeric(p["draft_round"], errors="coerce")
    p["draft_pick"] = pd.to_numeric(p["draft_pick"], errors="coerce")
    p["name_key"] = p["display_name"].map(_normalize_name)

    # Start from player master; it already contains draft capital and IDs.
    rookies = p[
        p["position"].isin(SKILL_POSITIONS)
        & p["draft_year"].notna()
    ].copy()

    # Merge combine by pfr_id first.
    c_by_id = c[c.get("pfr_id", pd.Series(index=c.index, dtype=object)).notna()].copy()
    if "pfr_id" in rookies.columns and not c_by_id.empty:
        cols = [
            "pfr_id", "season", "height_combine", "wt", "forty", "bench",
            "vertical", "broad_jump", "cone", "shuttle",
        ]
        c_by_id = c_by_id[[x for x in cols if x in c_by_id.columns]].drop_duplicates("pfr_id", keep="last")
        rookies = rookies.merge(c_by_id, on="pfr_id", how="left", suffixes=("", "_combine"))
    else:
        for col in ["height_combine", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]:
            rookies[col] = np.nan

    # Fallback combine merge by normalized name + draft year.
    combine_name = c[
        ["name_key", "season", "height_combine", "wt", "forty", "bench",
         "vertical", "broad_jump", "cone", "shuttle"]
    ].drop_duplicates(["name_key", "season"], keep="last")

    rookies = rookies.merge(
        combine_name,
        left_on=["name_key", "draft_year"],
        right_on=["name_key", "season"],
        how="left",
        suffixes=("", "_name"),
    )

    for col in ["height_combine", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]:
        alt = f"{col}_name"
        if alt in rookies.columns:
            rookies[col] = rookies[col].combine_first(rookies[alt])

    rookies["height"] = rookies["height"].combine_first(rookies["height_combine"])
    rookies["weight"] = rookies["weight"].combine_first(rookies["wt"])

    # Use draft-table capital as a fallback.
    d_small = d[
        ["season", "round", "pick", "position", "pfr_player_id", "pfr_player_name", "name_key"]
    ].copy()

    if "pfr_id" in rookies.columns:
        d_by_id = d_small[d_small["pfr_player_id"].notna()].drop_duplicates("pfr_player_id", keep="last")
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


def train_rookie_model(
    rookie_table: pd.DataFrame,
    stats: pd.DataFrame,
    target_season: int,
    config: TrainConfig,
):
    """Random forest is intentionally used here: rookie sample sizes are small."""
    stats = prepare_weekly_stats(stats)

    rookie_targets = (
        stats.groupby(["player_id", "season"], as_index=False)
        .agg(
            target_points=("fantasy_points_model", "sum"),
            games=("week", "nunique"),
        )
    )

    hist = rookie_table[rookie_table["draft_year"] < target_season].copy()
    hist["draft_year"] = pd.to_numeric(hist["draft_year"], errors="coerce")

    hist = hist.merge(
        rookie_targets,
        left_on=["gsis_id", "draft_year"],
        right_on=["player_id", "season"],
        how="inner",
    )

    hist = hist[hist["games"] >= 1].copy()
    features = ["position_id"] + ROOKIE_NUMERIC_FEATURES

    X = hist[features]
    y = hist["target_points"].astype(float)

    model = Pipeline(
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
    model.fit(X, y)
    return model, len(hist)


def project_rookies(
    rookie_model: Pipeline,
    rookie_table: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    current = rookie_table[
        rookie_table["draft_year"].eq(target_season)
        & rookie_table["position"].isin(SKILL_POSITIONS)
    ].copy()

    if current.empty:
        return pd.DataFrame()

    features = ["position_id"] + ROOKIE_NUMERIC_FEATURES
    X = current[features]

    # Pipeline point prediction.
    pred = rookie_model.predict(X)

    # Estimate uncertainty from the individual forest trees after imputation.
    imputer = rookie_model.named_steps["imputer"]
    forest = rookie_model.named_steps["model"]
    Xt = imputer.transform(X)
    tree_predictions = np.vstack([tree.predict(Xt) for tree in forest.estimators_])
    std = tree_predictions.std(axis=0)

    return pd.DataFrame(
        {
            "player_id": current["gsis_id"].astype(str).values,
            "player": current["display_name"].astype(str).values,
            "position": current["position"].astype(str).values,
            "projected_points": np.maximum(pred, 0.0),
            "uncertainty": np.maximum(std, 8.0),
            "rookie": True,
            "model": "rookie_prior",
            "draft_pick": current["draft_pick"].values,
            "draft_round": current["draft_round"].values,
        }
    )


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
    for pos, rank in replacement_ranks.items():
        vals = (
            df[df["position"].eq(pos)]["projected_points"]
            .sort_values(ascending=False)
            .to_numpy()
        )
        if len(vals) == 0:
            replacement[pos] = 0.0
        else:
            replacement[pos] = float(vals[min(rank - 1, len(vals) - 1)])

    df["replacement_points"] = df["position"].map(replacement).fillna(0.0)
    df["vorp"] = df["projected_points"] - df["replacement_points"]

    # Approximate 80% predictive interval.
    df["floor"] = np.maximum(0.0, df["projected_points"] - 1.28 * df["uncertainty"])
    df["ceiling"] = df["projected_points"] + 1.28 * df["uncertainty"]

    df = df.sort_values(
        ["vorp", "projected_points"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df.insert(0, "overall_rank", np.arange(1, len(df) + 1))

    df["position_rank"] = (
        df.groupby("position").cumcount() + 1
    )
    df["pos_rank"] = df["position"] + df["position_rank"].astype(str)
    return df


def save_artifacts(
    output_dir: Path,
    target_season: int,
    scoring: FantasyScoring,
    config: TrainConfig,
    model: FantasyTransformer,
    seq_scaler,
    static_imputer,
    static_scaler,
    rookie_model,
    metrics: dict,
    rankings: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{target_season}_ppr{scoring.reception:g}"

    torch_path = output_dir / f"fantasy_transformer_{tag}.pt"
    sklearn_path = output_dir / f"preprocessing_and_rookies_{tag}.joblib"
    rankings_path = output_dir / f"draft_rankings_{tag}.csv"
    metadata_path = output_dir / f"metadata_{tag}.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "weekly_features": WEEKLY_FEATURES,
            "static_features": STATIC_FEATURES,
        },
        torch_path,
    )

    joblib.dump(
        {
            "seq_scaler": seq_scaler,
            "static_imputer": static_imputer,
            "static_scaler": static_scaler,
            "rookie_model": rookie_model,
        },
        sklearn_path,
    )

    rankings.to_csv(rankings_path, index=False)

    metadata = {
        "target_season": target_season,
        "scoring": asdict(scoring),
        "config": asdict(config),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "transformer": str(torch_path),
        "preprocessing": str(sklearn_path),
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
    End-to-end function called by Streamlit.

    Important leakage rule:
    - model training only uses target seasons < requested target_season
    - each sample's input uses weeks strictly before its target season
    - requested target_season rankings use only prior-season NFL history
    """
    config = config or TrainConfig()
    scoring = scoring or FantasyScoring()
    output_dir = Path(output_dir)

    set_seed(config.random_seed)

    stats, players, combine, draft = load_sportsdataverse(
        start_season=config.start_season,
        end_season=target_season,
        scoring=scoring,
        source=source,
        progress=progress,
    )
    stats = prepare_weekly_stats(stats)

    # The requested target season is inference-only.
    historical_target_seasons = sorted(
        int(x)
        for x in stats["season"].dropna().unique()
        if config.start_season + 1 <= int(x) < target_season
    )

    if len(historical_target_seasons) < 3:
        raise ValueError(
            "Not enough historical seasons. Choose an earlier start season or "
            "a later target season."
        )

    if progress:
        progress("Building leakage-safe veteran player-season training sequences...")

    samples = build_veteran_training_samples(
        stats,
        players,
        historical_target_seasons,
        config,
    )

    if progress:
        progress(f"Training Transformer on {len(samples):,} veteran player-season samples...")

    (
        model,
        seq_scaler,
        static_imputer,
        static_scaler,
        target_mean,
        target_std,
        metrics,
        history,
    ) = train_transformer(samples, config, progress=progress)

    if progress:
        progress(f"Projecting veteran players for {target_season}...")

    veteran_rankings = project_veterans(
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

    if progress:
        progress("Training rookie prior from historical draft/combine classes...")

    rookie_table = prepare_rookie_table(players, combine, draft)
    rookie_model, rookie_training_samples = train_rookie_model(
        rookie_table,
        stats,
        target_season,
        config,
    )
    rookie_rankings = project_rookies(
        rookie_model,
        rookie_table,
        target_season,
    )

    veteran_diagnostics = dict(veteran_rankings.attrs)

    rankings = pd.concat(
        [veteran_rankings, rookie_rankings],
        ignore_index=True,
        sort=False,
    )
    rankings = add_draft_value(rankings, config)

    metrics["rookie_training_samples"] = rookie_training_samples
    metrics["ranked_veterans"] = int((~rankings["rookie"]).sum()) if not rankings.empty else 0
    metrics["ranked_rookies"] = int(rankings["rookie"].sum()) if not rankings.empty else 0
    metrics.update(veteran_diagnostics)

    paths = save_artifacts(
        output_dir=output_dir,
        target_season=target_season,
        scoring=scoring,
        config=config,
        model=model,
        seq_scaler=seq_scaler,
        static_imputer=static_imputer,
        static_scaler=static_scaler,
        rookie_model=rookie_model,
        metrics=metrics,
        rankings=rankings,
    )

    if progress:
        progress("Training complete. Draft rankings are ready.")

    return {
        "rankings": rankings,
        "veteran_rankings": veteran_rankings,
        "rookie_rankings": rookie_rankings,
        "metrics": metrics,
        "history": history,
        "paths": paths,
    }
