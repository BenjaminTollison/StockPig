from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import math
import gc

import numpy as np
import pandas as pd
import streamlit as st

import predictor_core_gpu as predictor


# =============================================================================
# League configuration
# =============================================================================

SEASON_WEEKS = list(range(1, 14))

# League rules: these weeks require TWO winning SEC picks.
DOUBLE_PICK_WEEKS = {1, 6, 7}

TRAIN_START_SEASON = predictor.TRAIN_START_SEASON
DEFAULT_LOOKBACK_GAMES = predictor.LOOKBACK_GAMES
PICKEM_MONTE_CARLO_SIMS = 2_000_000

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

TEAM_NAME_TO_ID = {
    name: team_id
    for team_id, (name, _) in SEC_TEAMS.items()
}


# =============================================================================
# Lightweight schedule loading
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule_only(season: int) -> pd.DataFrame:
    raw = predictor._load_cfb_schedule([int(season)])
    return predictor.normalize_schedule(raw)


def required_picks_for_week(week: int) -> int:
    return 2 if int(week) in DOUBLE_PICK_WEEKS else 1


def picks_required_before_week(week: int) -> int:
    return sum(
        required_picks_for_week(w)
        for w in SEASON_WEEKS
        if w < int(week)
    )


# =============================================================================
# Probability generation
# =============================================================================

def build_weekly_win_probabilities(
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    bundle: predictor.ModelBundle,
    season: int,
    planning_start_week: int,
    lookback_games: int,
    progress_callback=None,
) -> pd.DataFrame:
    """Build all future matchup features once, then batch predict/simulate them."""
    context = predictor.make_feature_context(pbp)
    rows: list[dict] = []
    feature_rows: list[dict] = []
    game_records: list[dict] = []

    weeks = [
        week
        for week in SEASON_WEEKS
        if week >= int(planning_start_week)
    ]
    total_weeks = max(len(weeks), 1)

    # CPU feature engineering remains the right fit here, but we build every
    # row first so sklearn inference can happen in one batched call.
    for week_number, week in enumerate(weeks, start=1):
        games = predictor.get_sec_week_games(schedule, int(season), int(week))

        for game in games.itertuples(index=False):
            if bool(game.completed):
                continue

            home_id = int(game.home_id)
            away_id = int(game.away_id)

            home_features = predictor.build_feature_row(
                context,
                offense_team=home_id,
                defense_team=away_id,
                is_home=True,
                season=int(season),
                week=int(planning_start_week),
                lookback_games=int(lookback_games),
            )
            away_features = predictor.build_feature_row(
                context,
                offense_team=away_id,
                defense_team=home_id,
                is_home=False,
                season=int(season),
                week=int(planning_start_week),
                lookback_games=int(lookback_games),
            )

            game_records.append(
                {
                    "week": int(week),
                    "game": game,
                    "home_id": home_id,
                    "away_id": away_id,
                }
            )
            feature_rows.extend((home_features, away_features))

        if progress_callback is not None:
            progress_callback(
                int(45 * week_number / total_weeks),
                f"Building future matchup features: Week {week} ({week_number}/{total_weeks})",
            )

    if not game_records:
        return pd.DataFrame(
            columns=[
                "week", "team_id", "team", "opponent_id", "opponent",
                "location", "game_id", "win_probability",
                "expected_points", "opponent_expected_points",
            ]
        )

    if progress_callback is not None:
        progress_callback(50, f"Batch-predicting {len(game_records):,} future games...")

    X = pd.DataFrame(feature_rows, columns=predictor.FEATURE_COLUMNS)
    predicted = np.maximum(bundle.model.predict(X).astype(float), 0.0).reshape(-1, 2)
    expected_home = predicted[:, 0]
    expected_away = predicted[:, 1]

    if progress_callback is not None:
        device = predictor.accelerator_name()
        progress_callback(65, f"Simulating {len(game_records):,} games on {device}...")

    # One matrix-shaped Monte Carlo call uses the RX 9070 XT efficiently and
    # avoids a separate GPU kernel launch for every game.
    home_scores, away_scores = predictor.simulate_games_from_residuals(
        expected_home=expected_home,
        expected_away=expected_away,
        bundle=bundle,
        n=PICKEM_MONTE_CARLO_SIMS,
        seed=(predictor.RANDOM_SEED + int(season) * 100_003 + int(planning_start_week)),
    )
    home_win, away_win, tie_prob = predictor.game_probabilities_batch(
        home_scores,
        away_scores,
    )

    if progress_callback is not None:
        progress_callback(90, "Formatting weekly SEC win probabilities...")

    for index, record in enumerate(game_records):
        game = record["game"]
        week = record["week"]
        home_id = record["home_id"]
        away_id = record["away_id"]

        home_probability = float(home_win[index] + 0.5 * tie_prob[index])
        away_probability = float(away_win[index] + 0.5 * tie_prob[index])

        if home_id in SEC_TEAMS:
            rows.append(
                {
                    "week": week,
                    "team_id": home_id,
                    "team": SEC_TEAMS[home_id][0],
                    "opponent_id": away_id,
                    "opponent": (
                        SEC_TEAMS[away_id][0]
                        if away_id in SEC_TEAMS
                        else str(game.away_team)
                    ),
                    "location": "Home",
                    "game_id": int(game.game_id),
                    "win_probability": home_probability,
                    "expected_points": float(expected_home[index]),
                    "opponent_expected_points": float(expected_away[index]),
                }
            )

        if away_id in SEC_TEAMS:
            rows.append(
                {
                    "week": week,
                    "team_id": away_id,
                    "team": SEC_TEAMS[away_id][0],
                    "opponent_id": home_id,
                    "opponent": (
                        SEC_TEAMS[home_id][0]
                        if home_id in SEC_TEAMS
                        else str(game.home_team)
                    ),
                    "location": "Away",
                    "game_id": int(game.game_id),
                    "win_probability": away_probability,
                    "expected_points": float(expected_away[index]),
                    "opponent_expected_points": float(expected_home[index]),
                }
            )

    if progress_callback is not None:
        progress_callback(100, "Future schedule simulation complete.")

    return (
        pd.DataFrame(rows)
        .sort_values(["week", "win_probability", "team"], ascending=[True, False, True])
        .reset_index(drop=True)
    )



# =============================================================================
# Season optimizer
# =============================================================================

def optimize_pickem_plan(
    probabilities: pd.DataFrame,
    planning_start_week: int,
    already_used_team_ids: set[int],
) -> tuple[float, list[tuple[int, tuple[int, ...]]]]:
    """Maximize probability of surviving every remaining required pick.

    Since survival probability is the product of selected team win
    probabilities, maximizing it is equivalent to maximizing:

        sum(log(p_team_win))

    The dynamic program enforces:
      * one or two picks according to league week rules,
      * each SEC team at most once,
      * no selecting both sides of the same game in a double-pick week,
      * only teams that actually have a game that week.
    """

    weeks = [
        week
        for week in SEASON_WEEKS
        if week >= int(planning_start_week)
    ]

    team_ids = sorted(SEC_TEAMS)
    bit_for_team = {
        team_id: 1 << index
        for index, team_id in enumerate(team_ids)
    }

    initial_used_mask = 0
    for team_id in already_used_team_ids:
        if team_id in bit_for_team:
            initial_used_mask |= bit_for_team[team_id]

    # Precompute every legal choice for each week.
    # Each option is: (team_mask, log_probability, tuple(team_ids))
    week_options: dict[int, list[tuple[int, float, tuple[int, ...]]]] = {}

    for week in weeks:
        week_rows = probabilities[
            probabilities["week"] == int(week)
        ].copy()

        # A team should have at most one game in a week. If the source somehow
        # contains duplicates, keep the highest probability row.
        week_rows = (
            week_rows
            .sort_values("win_probability", ascending=False)
            .drop_duplicates("team_id")
        )

        candidates = week_rows.to_dict("records")
        required = required_picks_for_week(week)

        options = []

        for combo in combinations(candidates, required):
            ids = tuple(int(row["team_id"]) for row in combo)

            # Double-pick week: never choose opposite sides of the same game.
            if required == 2:
                game_ids = [int(row["game_id"]) for row in combo]
                if len(set(game_ids)) != 2:
                    continue

            mask = 0
            log_probability = 0.0

            for row in combo:
                team_id = int(row["team_id"])
                probability = max(
                    min(float(row["win_probability"]), 1.0),
                    1e-9,
                )
                mask |= bit_for_team[team_id]
                log_probability += math.log(probability)

            options.append(
                (
                    mask,
                    log_probability,
                    tuple(sorted(ids)),
                )
            )

        week_options[week] = options

    @lru_cache(maxsize=None)
    def solve(
        week_index: int,
        used_mask: int,
    ) -> tuple[float, tuple[tuple[int, tuple[int, ...]], ...]]:
        if week_index >= len(weeks):
            return 0.0, tuple()

        week = weeks[week_index]

        best_score = -math.inf
        best_path: tuple[tuple[int, tuple[int, ...]], ...] = tuple()

        for option_mask, option_logp, team_tuple in week_options.get(week, []):
            if option_mask & used_mask:
                continue

            future_score, future_path = solve(
                week_index + 1,
                used_mask | option_mask,
            )

            if not math.isfinite(future_score):
                continue

            total_score = option_logp + future_score

            if total_score > best_score:
                best_score = total_score
                best_path = (
                    (int(week), team_tuple),
                    *future_path,
                )

        return best_score, best_path

    best_log_probability, path = solve(
        0,
        initial_used_mask,
    )

    if not math.isfinite(best_log_probability):
        raise ValueError(
            "No feasible pick plan exists with the selected already-used teams. "
            "A remaining week may not have enough unused SEC teams playing, or "
            "too many teams may already be marked as used."
        )

    return float(math.exp(best_log_probability)), list(path)


def recommendation_table(
    plan: list[tuple[int, tuple[int, ...]]],
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    cumulative_probability = 1.0

    lookup = probabilities.set_index(
        ["week", "team_id"],
        drop=False,
    )

    for week, team_ids in plan:
        selected = [
            lookup.loc[(int(week), int(team_id))]
            for team_id in team_ids
        ]

        week_probability = float(
            np.prod(
                [
                    float(row["win_probability"])
                    for row in selected
                ]
            )
        )
        cumulative_probability *= week_probability

        rows.append(
            {
                "Week": int(week),
                "Picks Required": required_picks_for_week(week),
                "Recommended Pick(s)": " + ".join(
                    str(row["team"])
                    for row in selected
                ),
                "Opponent(s)": " + ".join(
                    (
                        f'{row["opponent"]} '
                        f'({"vs" if row["location"] == "Home" else "@"})'
                    )
                    for row in selected
                ),
                "Win Probability": " + ".join(
                    f'{100 * float(row["win_probability"]):.1f}%'
                    for row in selected
                ),
                "Week Survival %": 100 * week_probability,
                "Cumulative Survival %": 100 * cumulative_probability,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["Week Survival %"] = result["Week Survival %"].round(2)
        result["Cumulative Survival %"] = result[
            "Cumulative Survival %"
        ].round(2)

    return result


# =============================================================================
# Streamlit page
# =============================================================================

st.title("🏆 SEC Fantasy Pick'em Optimizer")

if predictor.GPU_ACCELERATION_AVAILABLE:
    st.caption(f"⚡ Monte Carlo accelerator: {predictor.accelerator_name()} (ROCm/PyTorch)")
else:
    st.caption("Monte Carlo accelerator: CPU fallback")

st.caption(
    "Choose the sequence of SEC teams that maximizes the model-estimated "
    "probability of surviving the remaining regular season."
)

st.info(
    "League rule encoded here: one SEC team per week, except Weeks 1, 6, and 7 "
    "require TWO picks. A team may only be used once all season."
)

season = st.number_input(
    "Season",
    min_value=2024,
    max_value=predictor.DEFAULT_SEASON + 1,
    value=predictor.DEFAULT_SEASON,
    step=1,
)

with st.spinner("Loading the season schedule..."):
    schedule_preview = load_schedule_only(int(season))

try:
    current_week = predictor.get_default_week(
        schedule_preview,
        int(season),
    )
except ValueError:
    current_week = 1

current_week = min(max(int(current_week), 1), 13)

planning_start_week = st.selectbox(
    "Plan beginning with",
    options=[
        week
        for week in SEASON_WEEKS
        if week >= current_week
    ],
    index=0,
    format_func=lambda week: (
        f"Week {week}"
        + (" — TWO PICKS" if week in DOUBLE_PICK_WEEKS else "")
    ),
)

team_names = [
    SEC_TEAMS[team_id][0]
    for team_id in sorted(
        SEC_TEAMS,
        key=lambda tid: SEC_TEAMS[tid][0],
    )
]

already_used_names = st.multiselect(
    "SEC teams already used this season",
    options=team_names,
    help=(
        "These teams are removed from every future recommendation because "
        "the league only allows each team to be selected once."
    ),
)

already_used_ids = {
    TEAM_NAME_TO_ID[name]
    for name in already_used_names
}

expected_previous_picks = picks_required_before_week(
    int(planning_start_week)
)

if len(already_used_ids) != expected_previous_picks:
    st.warning(
        f"Entering Week {int(planning_start_week)}, the league format normally "
        f"implies {expected_previous_picks} team selections have already been "
        f"used. You currently selected {len(already_used_ids)}. The optimizer "
        f"will use your selected list exactly as entered."
    )

lookback_games = st.slider(
    "Recent games used to build team features",
    min_value=3,
    max_value=15,
    value=DEFAULT_LOOKBACK_GAMES,
)

remaining_slots = sum(
    required_picks_for_week(week)
    for week in SEASON_WEEKS
    if week >= int(planning_start_week)
)

remaining_teams = len(SEC_TEAMS) - len(already_used_ids)

c1, c2, c3 = st.columns(3)
c1.metric("Remaining pick slots", remaining_slots)
c2.metric("Unused SEC teams", remaining_teams)
c3.metric(
    "Double-pick weeks remaining",
    sum(
        1
        for week in DOUBLE_PICK_WEEKS
        if week >= int(planning_start_week)
    ),
)

if remaining_teams < remaining_slots:
    st.error(
        "There are fewer unused SEC teams than required remaining pick slots. "
        "Remove one or more teams from the already-used list."
    )
    st.stop()

# Older versions retained complete RandomForestRegressor bundles in
# session_state. Drop them immediately when this version is loaded.
legacy_model_cache = st.session_state.pop("_pickem_model_cache", None)
legacy_result = st.session_state.get("_pickem_result")
if isinstance(legacy_result, dict) and "bundle" in legacy_result:
    st.session_state.pop("_pickem_result", None)
if legacy_model_cache is not None:
    del legacy_model_cache
gc.collect()
try:
    if predictor.torch is not None and predictor.torch.cuda.is_available():
        predictor.torch.cuda.empty_cache()
except Exception:
    pass


run_key = (
    int(season),
    int(planning_start_week),
    int(lookback_games),
    tuple(sorted(already_used_ids)),
)

execute = st.button(
    "Execute training & optimize picks",
    type="primary",
    use_container_width=True,
)

if execute:
    st.session_state["_pickem_active_key"] = run_key

    data_progress = st.progress(
        0,
        text="Historical features: preparing...",
    )

    def update_data_progress(value: int, message: str) -> None:
        data_progress.progress(
            min(max(int(value), 0), 100),
            text=message,
        )

    with st.spinner("Loading SportsDataverse play-by-play..."):
        pbp, schedule = predictor.load_model_data(
            TRAIN_START_SEASON,
            int(season),
        )

    training_rows, feature_store_path, appended_feature_rows = (
        predictor.load_incremental_historical_features(
            first_training_season=TRAIN_START_SEASON,
            target_season=int(season),
            target_week=int(planning_start_week),
            lookback_games=int(lookback_games),
            progress_callback=update_data_progress,
        )
    )

    data_progress.progress(
        100,
        text=(
            f"Historical features ready • {len(training_rows):,} rows • "
            f"{appended_feature_rows:,} new rows persisted"
        ),
    )
    st.caption(f"Persistent feature store: `{feature_store_path}`")

    model_progress = st.progress(
        0,
        text="Model training/validation: preparing...",
    )

    def update_model_progress(value: int, message: str) -> None:
        model_progress.progress(
            min(max(int(value), 0), 100),
            text=message,
        )

    # Train only for this explicit optimization request. The completed
    # RandomForestRegressor is never stored in session_state.
    bundle = predictor.train_and_validate_model(
        training_rows,
        progress_callback=update_model_progress,
    )

    model_progress.progress(
        100,
        text=(
            f"Model ready • MAE {bundle.mae:.2f} pts • "
            "forest will be released after probabilities are generated"
        ),
    )

    validation_metrics = {
        "mae": float(bundle.mae),
        "rmse": float(bundle.rmse),
        "r2": float(bundle.r2),
        "validation_rows": int(bundle.validation_rows),
    }
    training_rows_count = int(len(training_rows))

    probability_progress = st.progress(
        0,
        text="Calculating win probabilities for the remaining schedule...",
    )

    def update_probability_progress(
        value: int,
        message: str,
    ) -> None:
        probability_progress.progress(
            min(max(int(value), 0), 100),
            text=message,
        )

    probabilities = build_weekly_win_probabilities(
        pbp=pbp,
        schedule=schedule,
        bundle=bundle,
        season=int(season),
        planning_start_week=int(planning_start_week),
        lookback_games=int(lookback_games),
        progress_callback=update_probability_progress,
    )

    probability_progress.progress(
        100,
        text=(
            f"Weekly win probabilities complete • "
            f"{len(probabilities):,} SEC team-game probabilities"
        ),
    )

    try:
        season_survival, plan = optimize_pickem_plan(
            probabilities=probabilities,
            planning_start_week=int(planning_start_week),
            already_used_team_ids=already_used_ids,
        )

        recommendations = recommendation_table(
            plan,
            probabilities,
        )

        # Persist only the small outputs needed to redraw the page. The
        # Random Forest, historical training frame, raw PBP, and schedule are
        # deliberately excluded.
        st.session_state["_pickem_result"] = {
            "key": run_key,
            "probabilities": probabilities,
            "recommendations": recommendations,
            "season_survival": float(season_survival),
            "training_rows_count": training_rows_count,
            "validation_metrics": validation_metrics,
            "feature_store_path": str(feature_store_path),
        }

    except ValueError as exc:
        st.session_state.pop("_pickem_result", None)
        st.error(str(exc))

    # Release the production forest, its validation data, the historical
    # feature DataFrame, and raw SportsDataverse frames as soon as all model
    # outputs have been materialized.
    del bundle
    del training_rows
    del pbp
    del schedule

    try:
        predictor.load_model_data.clear()
    except Exception:
        pass

    gc.collect()
    try:
        if predictor.torch is not None and predictor.torch.cuda.is_available():
            predictor.torch.cuda.empty_cache()
    except Exception:
        pass


result = st.session_state.get("_pickem_result")

if result is not None and result.get("key") == run_key:
    recommendations = result["recommendations"]
    probabilities = result["probabilities"]
    season_survival = float(result["season_survival"])
    metrics = result["validation_metrics"]

    st.divider()
    st.subheader("Optimal remaining pick plan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Estimated survival",
        f"{100 * season_survival:.2f}%",
    )
    m2.metric(
        "Validation MAE",
        f"{metrics['mae']:.2f} pts",
    )
    m3.metric(
        "Validation RMSE",
        f"{metrics['rmse']:.2f} pts",
    )
    m4.metric(
        "Training rows",
        f'{int(result["training_rows_count"]):,}',
    )

    st.caption(
        f"Persistent feature store: `{result['feature_store_path']}` • "
        "Random Forest released after probability generation"
    )

    st.dataframe(
        recommendations,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "The optimizer maximizes the product of the selected team win "
        "probabilities across the remaining season. On two-pick weeks, both "
        "teams must win, so that week's survival probability is the product "
        "of the two modeled win probabilities."
    )

    with st.expander("All modeled SEC win probabilities by week"):
        display = probabilities.copy()
        display["Win %"] = (
            100 * display["win_probability"]
        ).round(1)
        display["Expected Score"] = display[
            "expected_points"
        ].round(1)
        display["Opponent Expected"] = display[
            "opponent_expected_points"
        ].round(1)

        st.dataframe(
            display[
                [
                    "week",
                    "team",
                    "opponent",
                    "location",
                    "Win %",
                    "Expected Score",
                    "Opponent Expected",
                ]
            ].rename(
                columns={
                    "week": "Week",
                    "team": "Team",
                    "opponent": "Opponent",
                    "location": "Location",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Why this is not a greedy picker"):
        st.markdown(
            r"""
The page does **not** simply take the largest favorite every week.

Using a team this week removes it from every later week. The optimizer therefore
searches the remaining season jointly and may save a very strong team for a
future week where the alternatives are much worse.

If the selected weekly probabilities are $p_1, p_2, \ldots, p_n$, the
estimated probability of surviving the whole plan is:

$$
P(\text{survive}) = \prod_{i=1}^{n} p_i
$$

The optimizer works with logarithms for numerical stability:

$$
\max \sum_{i=1}^{n} \log(p_i)
$$

This is mathematically equivalent to maximizing the product above.
            """
        )

    st.warning(
        "The season-survival calculation treats outcomes in different games "
        "as independent. The Monte Carlo model captures prediction-error "
        "uncertainty within each game, but it does not model cross-game "
        "correlation such as shared weather or conference-wide shocks."
    )

elif not execute:
    st.caption(
        "Set the already-used teams and lookback window, then click "
        "**Execute training & optimize picks**. Historical features are read "
        "from the persistent Parquet store when available. The Random Forest "
        "is trained only for the explicit optimization run and is released "
        "immediately after the probability table is produced."
    )
