"""
Sleeper Draft Assistant

Supports:
1. A league-connected Sleeper draft
2. A standalone Sleeper mock draft using the draft ID / draftboard URL

Sleeper's public API is read-only, so this page watches the draft and recommends
players; you still make the actual pick in Sleeper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fantasy_football.draft_engine import (
    assign_roster_to_slots,
    available_players,
    next_pick_for_slot,
    normalize_rankings,
    pick_display_name,
    pick_position_for_overall,
    picks_for_draft_slot,
    roster_positions_from_draft,
    score_draft_board,
)
from fantasy_football.sleeper_api import (
    extract_draft_id,
    get_draft,
    get_draft_picks,
    get_league,
    get_league_drafts,
)


st.set_page_config(
    page_title="Sleeper Draft Assistant",
    page_icon="🏈",
    layout="wide",
)


def scoring_label_from_league(league: dict | None) -> str:
    if not league:
        return "Model scoring"

    reception = float(
        (league.get("scoring_settings") or {}).get("rec", 0.0)
    )

    if reception == 1.0:
        return "PPR"
    if reception == 0.5:
        return "Half-PPR"
    if reception == 0.0:
        return "Standard"
    return f"{reception:g} PPR"


def scoring_label_from_draft(draft: dict) -> str:
    metadata = draft.get("metadata") or {}
    scoring = str(
        metadata.get("scoring_type")
        or metadata.get("scoring")
        or ""
    ).lower()

    mapping = {
        "ppr": "PPR",
        "half_ppr": "Half-PPR",
        "half-ppr": "Half-PPR",
        "half": "Half-PPR",
        "standard": "Standard",
        "std": "Standard",
    }

    return mapping.get(scoring, "Model scoring")


def league_type_label(league: dict | None, draft: dict | None) -> str:
    if league:
        league_type = (league.get("settings") or {}).get("type")
        mapping = {
            0: "Redraft",
            1: "Keeper",
            2: "Dynasty",
        }

        try:
            return mapping.get(int(league_type), "Fantasy Draft")
        except (TypeError, ValueError):
            pass

    metadata = (draft or {}).get("metadata") or {}
    name = str(metadata.get("name") or "").lower()

    if "dynasty" in name:
        return "Dynasty"
    if "keeper" in name:
        return "Keeper"

    return "Redraft"


def saved_ranking_files() -> list[Path]:
    models = Path("models")
    if not models.exists():
        return []

    return sorted(
        models.glob("draft_rankings_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_rankings_from_ui() -> tuple[pd.DataFrame, str]:
    session_result = st.session_state.get("fantasy_result")
    saved_files = saved_ranking_files()

    source_options = []
    if isinstance(session_result, dict):
        source_options.append("Current trained rankings")
    if saved_files:
        source_options.append("Saved rankings CSV")
    source_options.append("Upload rankings CSV")

    source = st.sidebar.radio(
        "Rankings source",
        source_options,
        index=0,
    )

    if source == "Current trained rankings":
        frame = session_result.get("rankings")
        if isinstance(frame, pd.DataFrame):
            return normalize_rankings(frame), "Current trained rankings"

        st.sidebar.error(
            "The current training session has no rankings DataFrame."
        )
        return pd.DataFrame(), source

    if source == "Saved rankings CSV":
        labels = [path.name for path in saved_files]
        selected = st.sidebar.selectbox(
            "Saved rankings",
            labels,
        )
        selected_path = saved_files[labels.index(selected)]
        return (
            normalize_rankings(pd.read_csv(selected_path)),
            selected_path.name,
        )

    uploaded = st.sidebar.file_uploader(
        "Upload fantasy rankings CSV",
        type=["csv"],
    )

    if uploaded is None:
        return pd.DataFrame(), "No rankings loaded"

    return (
        normalize_rankings(pd.read_csv(uploaded)),
        uploaded.name,
    )


def load_league_draft(league_id: str) -> None:
    league = get_league(league_id)
    drafts = get_league_drafts(league_id)

    st.session_state["draft_assistant_league"] = league
    st.session_state["sleeper_league"] = league
    st.session_state["sleeper_league_id"] = str(league["league_id"])
    st.session_state["draft_assistant_drafts"] = drafts

    if not drafts:
        raise ValueError("No draft was found for this Sleeper league.")

    draft = drafts[0]
    draft_id = str(draft["draft_id"])

    st.session_state["draft_assistant_draft_id"] = draft_id
    st.session_state["draft_assistant_draft"] = get_draft(draft_id)
    st.session_state["draft_assistant_picks"] = get_draft_picks(draft_id)


def load_mock_draft(value: str) -> None:
    draft_id = extract_draft_id(value)
    draft = get_draft(draft_id)

    st.session_state["draft_assistant_draft_id"] = draft_id
    st.session_state["draft_assistant_draft"] = draft
    st.session_state["draft_assistant_picks"] = get_draft_picks(draft_id)

    # A standalone mock does not necessarily belong to a league.
    st.session_state["draft_assistant_league"] = None
    st.session_state["draft_assistant_drafts"] = []
    st.session_state["draft_assistant_mock_input"] = value


def refresh_draft_state() -> None:
    draft_id = st.session_state.get("draft_assistant_draft_id")
    if not draft_id:
        raise ValueError("Load a draft first.")

    st.session_state["draft_assistant_draft"] = get_draft(draft_id)
    st.session_state["draft_assistant_picks"] = get_draft_picks(draft_id)


st.title("🏈 Sleeper Draft Assistant")
st.caption(
    "Use the model rankings beside a live league draft or a Sleeper mock draft."
)

rankings, rankings_source = load_rankings_from_ui()

with st.sidebar:
    st.divider()
    st.header("Sleeper Draft")

    mode = st.radio(
        "Draft source",
        ["League Draft", "Mock Draft / Draft ID"],
        horizontal=False,
    )

    if mode == "League Draft":
        league_id = st.text_input(
            "League ID",
            value=st.session_state.get("sleeper_league_id", ""),
            placeholder="1234567890",
        )

        if st.button(
            "Load league draft",
            use_container_width=True,
            type="primary",
        ):
            try:
                load_league_draft(league_id)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load league draft: {exc}")

    else:
        draft_input = st.text_input(
            "Mock draft ID or URL",
            value=st.session_state.get(
                "draft_assistant_mock_input",
                "",
            ),
            placeholder="https://sleeper.com/draft/nfl/...",
            help=(
                "Open the mock draftboard in a browser and paste its URL here. "
                "You can also paste only the numeric draft ID."
            ),
        )

        if st.button(
            "Connect to mock draft",
            use_container_width=True,
            type="primary",
        ):
            try:
                load_mock_draft(draft_input)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load mock draft: {exc}")

    if st.button(
        "Refresh live picks",
        use_container_width=True,
    ):
        try:
            refresh_draft_state()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not refresh draft: {exc}")


league = st.session_state.get("draft_assistant_league")
draft = st.session_state.get("draft_assistant_draft")
picks = st.session_state.get("draft_assistant_picks", [])

if not draft:
    st.info(
        "Load a Sleeper league draft or connect directly to a mock draftboard."
    )
    st.stop()

if rankings.empty:
    st.warning(
        "Load a rankings source in the sidebar before using the draft assistant."
    )
    st.stop()


draft_settings = draft.get("settings") or {}

teams = int(
    draft_settings.get("teams")
    or (league or {}).get("total_rosters")
    or 12
)

rounds = int(draft_settings.get("rounds") or 15)
draft_type = str(draft.get("type") or "snake").lower()

# Standalone mock drafts do not have league.roster_positions, so reconstruct
# them from the draft's slot counts.
if league and league.get("roster_positions"):
    roster_positions = league["roster_positions"]
else:
    roster_positions = roster_positions_from_draft(draft)

if not roster_positions:
    roster_positions = [
        "QB",
        "RB",
        "RB",
        "WR",
        "WR",
        "TE",
        "FLEX",
        "BN",
        "BN",
        "BN",
        "BN",
        "BN",
    ]


# ------------------------------------------------------------------
# Draft slot
# ------------------------------------------------------------------
slot_default = int(
    st.session_state.get("draft_assistant_slot", 1)
)
slot_default = min(max(slot_default, 1), teams)

with st.sidebar:
    user_slot = st.selectbox(
        "Your draft slot",
        options=list(range(1, teams + 1)),
        index=slot_default - 1,
    )
    st.session_state["draft_assistant_slot"] = int(user_slot)
    st.caption(f"Rankings: {rankings_source}")

    status = str(draft.get("status") or "unknown")
    st.caption(f"Draft status: {status}")


# ------------------------------------------------------------------
# Draft header
# ------------------------------------------------------------------
draft_season = str(
    draft.get("season")
    or (league or {}).get("season")
    or ""
)

draft_kind = league_type_label(league, draft)

metadata = draft.get("metadata") or {}
draft_title = str(
    metadata.get("name")
    or f"{draft_season} {draft_kind}"
).strip()

if league:
    format_scoring = scoring_label_from_league(league)
else:
    format_scoring = scoring_label_from_draft(draft)

format_name = f"{teams}-team {format_scoring}"

completed_picks = len(picks)
max_pick = teams * rounds
current_pick_no = min(completed_picks + 1, max_pick)

current_round, current_within, current_draft_slot = (
    pick_position_for_overall(
        current_pick_no,
        teams,
        draft_type,
    )
)

next_user_pick = next_pick_for_slot(
    current_pick_no=current_pick_no,
    teams=teams,
    user_slot=int(user_slot),
    rounds=rounds,
    draft_type=draft_type,
)

header_cols = st.columns(4)
header_cols[0].metric(
    "Draft ID",
    str(draft.get("draft_id", "")),
)
header_cols[1].metric("Draft", draft_title)
header_cols[2].metric("Format", format_name)
header_cols[3].metric("Draft Slot", int(user_slot))

status_left, status_right = st.columns(2)

if completed_picks >= max_pick:
    status_left.subheader(
        f"Round {rounds} — Pick {max_pick}"
    )
    status_right.subheader("Draft complete")
    picks_until_next = None
else:
    status_left.subheader(
        f"Round {current_round} — Pick {current_pick_no}"
    )

    if next_user_pick is None:
        status_right.subheader("No remaining picks")
        picks_until_next = None
    else:
        picks_until_next = max(
            0,
            int(next_user_pick["pick_no"]) - current_pick_no,
        )

        if int(next_user_pick["pick_no"]) == current_pick_no:
            status_right.subheader(
                f"Your next pick: {next_user_pick['label']} — ON THE CLOCK"
            )
        else:
            status_right.subheader(
                f"Your next pick: {next_user_pick['label']} "
                f"(overall {next_user_pick['pick_no']})"
            )

st.divider()


# ------------------------------------------------------------------
# Roster
# ------------------------------------------------------------------
st.subheader("YOUR ROSTER")

all_player_names = rankings["player"].dropna().astype(str).tolist()

sleeper_my_picks = picks_for_draft_slot(
    picks,
    int(user_slot),
)
sleeper_my_names = [
    pick_display_name(pick)
    for pick in sleeper_my_picks
]

# In mock-draft mode, auto-sync is much more useful than manual selection.
default_sync = mode == "Mock Draft / Draft ID"

sync_from_sleeper = st.checkbox(
    "Sync my roster from Sleeper picks",
    value=default_sync,
    help=(
        "Uses the picks from your selected draft slot. This is recommended "
        "for mock drafts."
    ),
)

default_roster = st.session_state.get(
    "draft_assistant_manual_roster",
    [],
)

if sync_from_sleeper:
    ranked_name_map = {
        name.lower(): name
        for name in all_player_names
    }

    default_roster = [
        ranked_name_map[name.lower()]
        for name in sleeper_my_names
        if name.lower() in ranked_name_map
    ]

manual_roster = st.multiselect(
    "Your drafted players",
    options=all_player_names,
    default=[
        name
        for name in default_roster
        if name in all_player_names
    ],
    disabled=sync_from_sleeper,
    help=(
        "Turn off Sleeper sync if you want to override the roster manually."
    ),
)

if sync_from_sleeper:
    selected_roster = default_roster
else:
    selected_roster = manual_roster
    st.session_state[
        "draft_assistant_manual_roster"
    ] = manual_roster

roster_view, bench_players = assign_roster_to_slots(
    rankings,
    selected_roster,
    roster_positions,
)

roster_display = roster_view[
    ["slot", "player", "position", "projected_points"]
].copy()

roster_display["projected_points"] = roster_display[
    "projected_points"
].round(1)

st.dataframe(
    roster_display,
    use_container_width=True,
    hide_index=True,
)

if bench_players:
    st.caption("Bench: " + ", ".join(bench_players))

st.divider()


# ------------------------------------------------------------------
# Recommendation
# ------------------------------------------------------------------
available = available_players(
    rankings=rankings,
    picks=picks,
    manually_selected_roster=selected_roster,
)

recommendations = score_draft_board(
    available=available,
    roster_view=roster_view,
    teams=teams,
    picks_until_next_turn=picks_until_next,
)

st.subheader("RECOMMENDED PICK")

if recommendations.empty:
    st.info("No ranked players remain available.")
else:
    top = recommendations.iloc[0]

    rec_left, rec_right = st.columns([2, 1])

    with rec_left:
        st.markdown(
            f"### 1. {top['player']} · {top['position']}"
        )

        metric_cols = st.columns(3)

        metric_cols[0].metric(
            "Projection",
            f"{float(top['projected_points']):.1f}",
        )
        metric_cols[1].metric(
            "VORP",
            f"{float(top['vorp']):+.1f}",
        )
        metric_cols[2].metric(
            "Draft score",
            f"{float(top['draft_score']):.1f}",
        )

    with rec_right:
        st.markdown("**Why:**")
        for reason in top["reasons"]:
            st.markdown(f"• {reason}")

    st.markdown("#### Alternatives")

    alternatives = recommendations.iloc[1:4].copy()

    if not alternatives.empty:
        alternatives.insert(
            0,
            "recommendation",
            range(2, 2 + len(alternatives)),
        )

        alt_display = alternatives[
            [
                "recommendation",
                "player",
                "position",
                "draft_score",
                "projected_points",
                "vorp",
            ]
        ].rename(
            columns={
                "recommendation": "#",
                "player": "Player",
                "position": "Pos",
                "draft_score": "Draft score",
                "projected_points": "Proj",
                "vorp": "VORP",
            }
        )

        st.dataframe(
            alt_display.round(1),
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# ------------------------------------------------------------------
# Available players
# ------------------------------------------------------------------
st.subheader("AVAILABLE PLAYERS")

filter_col, count_col = st.columns([2, 1])

with filter_col:
    positions = st.multiselect(
        "Position filter",
        ["QB", "RB", "WR", "TE"],
        default=["QB", "RB", "WR", "TE"],
    )

with count_col:
    show_count = st.selectbox(
        "Rows",
        [25, 50, 100, 200],
        index=1,
    )

available_display = recommendations[
    recommendations["position"].isin(positions)
].head(int(show_count)).copy()

available_display.insert(
    0,
    "RK",
    range(1, len(available_display) + 1),
)

display_columns = [
    "RK",
    "player",
    "position",
    "projected_points",
    "vorp",
    "draft_score",
]

st.dataframe(
    available_display[display_columns]
    .rename(
        columns={
            "player": "PLAYER",
            "position": "POS",
            "projected_points": "PROJ",
            "vorp": "VORP",
            "draft_score": "DRAFT SCORE",
        }
    )
    .round(1),
    use_container_width=True,
    hide_index=True,
    height=650,
)


# ------------------------------------------------------------------
# Live board
# ------------------------------------------------------------------
with st.expander("Sleeper draft picks"):
    if not picks:
        st.caption("No picks have been made yet.")
    else:
        pick_rows = []

        for pick in picks:
            pick_rows.append(
                {
                    "Pick": pick.get("pick_no"),
                    "Round": pick.get("round"),
                    "Draft slot": pick.get("draft_slot"),
                    "Player": pick_display_name(pick),
                    "Pos": (pick.get("metadata") or {}).get(
                        "position"
                    ),
                    "Team": (pick.get("metadata") or {}).get(
                        "team"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(pick_rows),
            use_container_width=True,
            hide_index=True,
        )


with st.expander("Mock-draft instructions"):
    st.markdown(
        """
1. Start a Sleeper mock draft.
2. Open the draftboard in a browser.
3. Copy the URL. It will contain a numeric draft ID, for example:
   `sleeper.com/draft/nfl/1234567890123456789`.
4. Select **Mock Draft / Draft ID** in this page.
5. Paste the URL and click **Connect to mock draft**.
6. Select your draft slot.
7. Leave **Sync my roster from Sleeper picks** enabled.
8. Click **Refresh live picks** as the bots draft.

Sleeper's public API is read-only, so the assistant watches the mock but does
not make the Sleeper selection for you.
"""
    )
