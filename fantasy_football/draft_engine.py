"""
Pure draft-assistant logic.

This file intentionally contains no Streamlit calls and no HTTP calls.  That
makes it easier to test draft strategy independently from the UI.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

BENCH_LIKE_SLOTS = {
    "BN",
    "BENCH",
    "IR",
    "RESERVE",
    "TAXI",
}

SLOT_ELIGIBILITY = {
    "QB": {"QB"},
    "RB": {"RB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"WR", "RB"},
    "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}


def normalize_name(value: object) -> str:
    value = "" if value is None else str(value)
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a rankings DataFrame for the draft assistant."""
    if rankings is None or rankings.empty:
        return pd.DataFrame()

    df = rankings.copy()

    required = {"player", "position", "projected_points"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Rankings are missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["player"] = df["player"].astype(str)
    df["player_key"] = df["player"].map(normalize_name)
    df["position"] = df["position"].astype(str).str.upper()

    for column in [
        "projected_points",
        "vorp",
        "overall_rank",
        "position_rank",
        "floor",
        "ceiling",
        "uncertainty",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "vorp" not in df.columns:
        # A safe fallback if a CSV came from an older model version.
        replacement = {}
        fallback_ranks = {"QB": 12, "RB": 30, "WR": 36, "TE": 12}
        for position, rank in fallback_ranks.items():
            vals = (
                df.loc[df["position"].eq(position), "projected_points"]
                .dropna()
                .sort_values(ascending=False)
                .to_numpy()
            )
            replacement[position] = (
                float(vals[min(rank - 1, len(vals) - 1)])
                if len(vals)
                else 0.0
            )
        df["vorp"] = (
            df["projected_points"]
            - df["position"].map(replacement).fillna(0.0)
        )

    if "overall_rank" not in df.columns:
        df = df.sort_values(
            ["vorp", "projected_points"],
            ascending=[False, False],
        ).reset_index(drop=True)
        df["overall_rank"] = np.arange(1, len(df) + 1)

    if "position_rank" not in df.columns:
        position_order = (
            df.sort_values(
                ["position", "projected_points"],
                ascending=[True, False],
            )
            .groupby("position")
            .cumcount()
            + 1
        )
        df["position_rank"] = position_order

    if "pos_rank" not in df.columns:
        df["pos_rank"] = (
            df["position"] + df["position_rank"].fillna(0).astype(int).astype(str)
        )

    return df.reset_index(drop=True)


def roster_positions_from_draft(draft: dict) -> list[str]:
    """
    Reconstruct Sleeper roster positions for standalone/mock drafts.

    Standalone mock drafts may not have a league object, but their draft
    settings contain slot counts such as slots_qb, slots_rb, slots_wr, etc.
    """
    settings = draft.get("settings") or {}

    mapping = [
        ("slots_qb", "QB"),
        ("slots_rb", "RB"),
        ("slots_wr", "WR"),
        ("slots_te", "TE"),
        ("slots_flex", "FLEX"),
        ("slots_super_flex", "SUPER_FLEX"),
        ("slots_wrrb_flex", "WRRB_FLEX"),
        ("slots_rec_flex", "REC_FLEX"),
        ("slots_k", "K"),
        ("slots_def", "DEF"),
        ("slots_bn", "BN"),
    ]

    positions: list[str] = []

    for key, label in mapping:
        raw = settings.get(key, 0)
        try:
            count = int(raw or 0)
        except (TypeError, ValueError):
            count = 0

        positions.extend([label] * max(0, count))

    return positions


def starter_slot_labels(roster_positions: Iterable[str]) -> list[dict]:
    """
    Convert Sleeper roster positions into display labels such as RB1, RB2, WR1.

    Bench/IR-style slots are intentionally omitted from the starting-lineup
    display because the assistant's first goal is to construct the strongest
    starting roster.
    """
    counts: dict[str, int] = {}
    output: list[dict] = []

    for raw_slot in roster_positions:
        slot = str(raw_slot).upper()

        if slot in BENCH_LIKE_SLOTS:
            continue

        counts[slot] = counts.get(slot, 0) + 1
        count = counts[slot]

        # Number repeated base positions; keep a single FLEX/TE/QB label clean
        # unless that slot is repeated.
        repeat_total = sum(
            1 for value in roster_positions if str(value).upper() == slot
        )
        label = f"{slot}{count}" if repeat_total > 1 else slot

        output.append(
            {
                "slot": slot,
                "label": label,
                "eligible": SLOT_ELIGIBILITY.get(slot, {slot}),
            }
        )

    return output


def assign_roster_to_slots(
    rankings: pd.DataFrame,
    selected_players: list[str],
    roster_positions: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Greedily place selected players into starting slots, then return leftovers
    as bench players.

    Exact-position slots are filled before flex slots so an RB is not consumed
    by FLEX while an RB starter spot is still empty.
    """
    rankings = normalize_rankings(rankings)
    slots = starter_slot_labels(list(roster_positions))

    selected_df = rankings[
        rankings["player"].isin(selected_players)
    ].copy()

    # Preserve user selection order where possible.
    order = {name: idx for idx, name in enumerate(selected_players)}
    selected_df["_selected_order"] = selected_df["player"].map(order)
    selected_df = selected_df.sort_values("_selected_order")

    unassigned = selected_df.to_dict("records")
    rows = []

    # Exact/base slots first.
    base_indices = [
        idx
        for idx, slot in enumerate(slots)
        if slot["slot"] in SKILL_POSITIONS
    ]
    flex_indices = [
        idx
        for idx, slot in enumerate(slots)
        if idx not in base_indices
    ]

    assigned_by_index: dict[int, dict | None] = {}

    for idx in base_indices + flex_indices:
        slot = slots[idx]
        chosen_index = None

        for player_idx, player in enumerate(unassigned):
            if player["position"] in slot["eligible"]:
                chosen_index = player_idx
                break

        chosen = None
        if chosen_index is not None:
            chosen = unassigned.pop(chosen_index)

        assigned_by_index[idx] = chosen

    for idx, slot in enumerate(slots):
        player = assigned_by_index.get(idx)
        rows.append(
            {
                "slot": slot["label"],
                "slot_type": slot["slot"],
                "player": player["player"] if player else "—",
                "position": player["position"] if player else "",
                "projected_points": (
                    float(player["projected_points"])
                    if player
                    else np.nan
                ),
                "filled": player is not None,
            }
        )

    bench_players = [player["player"] for player in unassigned]

    return pd.DataFrame(rows), bench_players


def drafted_player_keys(picks: list[dict]) -> tuple[set[str], set[str]]:
    """Return picked Sleeper IDs and normalized player names."""
    ids: set[str] = set()
    names: set[str] = set()

    for pick in picks:
        player_id = pick.get("player_id")
        if player_id:
            ids.add(str(player_id))

        metadata = pick.get("metadata") or {}
        first = str(metadata.get("first_name") or "").strip()
        last = str(metadata.get("last_name") or "").strip()
        full = f"{first} {last}".strip()

        if full:
            names.add(normalize_name(full))

    return ids, names


def available_players(
    rankings: pd.DataFrame,
    picks: list[dict],
    manually_selected_roster: list[str] | None = None,
) -> pd.DataFrame:
    """Remove players already drafted in Sleeper or manually placed on roster."""
    df = normalize_rankings(rankings)
    if df.empty:
        return df

    picked_ids, picked_names = drafted_player_keys(picks)

    drafted_mask = df["player_key"].isin(picked_names)

    if "sleeper_id" in df.columns:
        sleeper_ids = df["sleeper_id"].astype(str)
        drafted_mask = drafted_mask | sleeper_ids.isin(picked_ids)

    manual = set(manually_selected_roster or [])
    if manual:
        drafted_mask = drafted_mask | df["player"].isin(manual)

    return df.loc[~drafted_mask].copy().reset_index(drop=True)


def pick_position_for_overall(
    pick_no: int,
    teams: int,
    draft_type: str = "snake",
) -> tuple[int, int, int]:
    """
    Return (round, pick-within-round, draft-slot) for an overall pick number.

    Sleeper's common types are snake and linear. Unknown types fall back to
    snake scheduling for the recommendation UI.
    """
    pick_no = max(1, int(pick_no))
    teams = max(1, int(teams))
    draft_type = str(draft_type or "snake").lower()

    round_no = ((pick_no - 1) // teams) + 1
    within = ((pick_no - 1) % teams) + 1

    if draft_type == "linear":
        draft_slot = within
    else:
        draft_slot = (
            within
            if round_no % 2 == 1
            else teams - within + 1
        )

    return round_no, within, draft_slot


def next_pick_for_slot(
    current_pick_no: int,
    teams: int,
    user_slot: int,
    rounds: int,
    draft_type: str = "snake",
) -> dict | None:
    """
    Find the user's next scheduled selection at or after current_pick_no.
    """
    max_pick = max(1, int(teams) * int(rounds))

    for pick_no in range(max(1, int(current_pick_no)), max_pick + 1):
        round_no, within, draft_slot = pick_position_for_overall(
            pick_no,
            teams,
            draft_type,
        )

        if int(draft_slot) == int(user_slot):
            return {
                "pick_no": pick_no,
                "round": round_no,
                "within_round": within,
                "draft_slot": draft_slot,
                "label": f"{round_no}.{within:02d}",
            }

    return None


def _position_level_reason(
    position: str,
    position_rank: float | int | None,
    teams: int,
) -> str | None:
    if position_rank is None or pd.isna(position_rank):
        return None

    rank = int(position_rank)
    teams = max(int(teams), 1)
    level = math.ceil(rank / teams)

    if level <= 2:
        return f"{position}{level}-level season projection"

    return None


def _open_slot_reason(
    player_position: str,
    roster_view: pd.DataFrame,
) -> tuple[float, str | None]:
    """
    Reward players that fill currently empty starting slots.

    Base position need is worth more than a flex-only need.
    """
    if roster_view.empty:
        return 0.0, None

    empty = roster_view[~roster_view["filled"]]

    exact = empty[empty["slot_type"].eq(player_position)]
    if not exact.empty:
        label = str(exact.iloc[0]["slot"])
        return 10.0, f"fills open {label} starter slot"

    for _, row in empty.iterrows():
        slot = str(row["slot_type"])
        eligible = SLOT_ELIGIBILITY.get(slot, {slot})
        if player_position in eligible:
            return 4.0, f"fills open {row['slot']} slot"

    return 0.0, None


def score_draft_board(
    available: pd.DataFrame,
    roster_view: pd.DataFrame,
    teams: int,
    picks_until_next_turn: int | None,
) -> pd.DataFrame:
    """
    Add a heuristic draft_score and human-readable recommendation reasons.

    This is deliberately separate from projected fantasy points.  VORP remains
    the base value; the draft score adds situational bonuses for roster need,
    positional tier drop, and the chance that waiting one turn loses the player.
    """
    board = normalize_rankings(available)
    if board.empty:
        return board

    board = board.sort_values(
        ["vorp", "projected_points"],
        ascending=[False, False],
    ).reset_index(drop=True)

    board["available_rank"] = np.arange(1, len(board) + 1)

    scores = []
    reasons_out = []
    need_bonuses = []
    tier_bonuses = []
    return_bonuses = []

    # Precompute position pools in projection order.
    position_pools = {
        position: (
            board[board["position"].eq(position)]
            .sort_values("projected_points", ascending=False)
            .reset_index(drop=True)
        )
        for position in SKILL_POSITIONS
    }

    for _, row in board.iterrows():
        position = str(row["position"])
        base_vorp = float(row.get("vorp", 0.0) or 0.0)
        reasons: list[str] = []

        level_reason = _position_level_reason(
            position,
            row.get("position_rank"),
            teams,
        )
        if level_reason:
            reasons.append(level_reason)

        need_bonus, need_reason = _open_slot_reason(
            position,
            roster_view,
        )
        if need_reason:
            reasons.append(need_reason)

        # How sharply does this position drop after the current player?
        tier_bonus = 0.0
        pool = position_pools.get(position, pd.DataFrame())
        if not pool.empty:
            matches = pool.index[
                pool["player_key"].eq(row["player_key"])
            ].tolist()

            if matches:
                idx = matches[0]
                lookahead_idx = min(idx + 4, len(pool) - 1)
                future_projection = float(
                    pool.iloc[lookahead_idx]["projected_points"]
                )
                projection_drop = max(
                    0.0,
                    float(row["projected_points"]) - future_projection,
                )
                tier_bonus = min(10.0, projection_drop * 0.20)

                if projection_drop >= 12.0 and lookahead_idx > idx:
                    count = lookahead_idx - idx
                    reasons.append(
                        f"large {position} tier drop over the next {count} options"
                    )

        # Crude "will he make it back?" heuristic based on the MODEL board.
        # It is intentionally labeled as a model-board signal, not market ADP.
        return_bonus = 0.0
        if (
            picks_until_next_turn is not None
            and picks_until_next_turn > 0
            and int(row["available_rank"]) <= picks_until_next_turn
        ):
            return_bonus = 5.0
            reasons.append(
                "model board suggests he may not return at your next pick"
            )

        draft_score = (
            base_vorp
            + need_bonus
            + tier_bonus
            + return_bonus
        )

        if not reasons:
            reasons.append("best remaining combination of projection and VORP")

        scores.append(draft_score)
        reasons_out.append(reasons[:4])
        need_bonuses.append(need_bonus)
        tier_bonuses.append(tier_bonus)
        return_bonuses.append(return_bonus)

    board["draft_score"] = scores
    board["reasons"] = reasons_out
    board["need_bonus"] = need_bonuses
    board["tier_bonus"] = tier_bonuses
    board["return_bonus"] = return_bonuses

    return board.sort_values(
        ["draft_score", "vorp", "projected_points"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def picks_for_draft_slot(
    picks: list[dict],
    draft_slot: int,
) -> list[dict]:
    return [
        pick
        for pick in picks
        if int(pick.get("draft_slot") or -1) == int(draft_slot)
    ]


def pick_display_name(pick: dict) -> str:
    metadata = pick.get("metadata") or {}
    first = str(metadata.get("first_name") or "").strip()
    last = str(metadata.get("last_name") or "").strip()
    name = f"{first} {last}".strip()
    return name or str(pick.get("player_id") or "Unknown player")
