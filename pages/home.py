import streamlit as st

st.title("🏈 StockPig")
st.subheader("Stock is a reference to Stockfish the chess bot")
st.subheader("Pig is a reference to pigskin")

st.markdown("""
StockPig uses college-football play-by-play data for two main jobs:

1. **Predict team scores and win probabilities**
2. **Rank SEC teams by offensive and defensive performance**

All features are calculated using games played **before the selected week** so
the model cannot use information from the game it is trying to predict.
""")

st.info(
    "Example: a Week 6 prediction may use Weeks 1–5 and previous-season games, "
    "but it cannot use Week 6 or later games."
)

st.header("1. Expected Points Added (EPA)")

st.markdown("""
Expected Points estimates how many points an offense is expected to score from
a particular situation. EPA measures how much one play changes that expectation.
""")

st.latex(r"EPA_{play} = EP_{after} - EP_{before}")

st.markdown("""
Positive EPA is good for the offense. Because defensive EPA is still measured
from the offense's perspective, **lower EPA allowed is better for the defense**.
""")

st.header("2. Offensive Features")

with st.expander("EPA / Play", expanded=True):
    st.latex(
        r"\text{Off EPA / Play} = "
        r"\frac{\sum EPA_{\text{offensive plays}}}{\text{Offensive plays}}"
    )

with st.expander("Pass EPA / Play"):
    st.latex(
        r"\text{Pass EPA / Play} = "
        r"\frac{\sum EPA_{\text{pass plays}}}{\text{Pass plays}}"
    )

with st.expander("Rush EPA / Play"):
    st.latex(
        r"\text{Rush EPA / Play} = "
        r"\frac{\sum EPA_{\text{rush plays}}}{\text{Rush plays}}"
    )

with st.expander("Success Rate"):
    st.latex(
        r"\text{Success Rate} = "
        r"\frac{\#(EPA > 0)}{\text{Offensive plays}}"
    )

with st.expander("Pass Rate"):
    st.latex(
        r"\text{Pass Rate} = "
        r"\frac{\text{Pass plays}}{\text{Pass plays}+\text{Rush plays}}"
    )
    st.caption("Displayed as offensive style; it is not part of the offensive composite score.")

with st.expander("Turnover Rate"):
    st.latex(
        r"\text{Turnover Rate} = "
        r"\frac{\text{Turnover plays}}{\text{Offensive plays}}"
    )
    st.caption("Lower is better.")

with st.expander("Points / Game"):
    st.latex(
        r"\text{Points / Game} = "
        r"\frac{\sum \text{Points scored}}{\text{Games}}"
    )

with st.expander("Plays / Game"):
    st.latex(
        r"\text{Plays / Game} = "
        r"\frac{\text{Offensive plays}}{\text{Games}}"
    )
    st.caption("Displayed as pace; it is not part of the offensive composite score.")

st.header("3. Defensive Features")

with st.expander("EPA Allowed / Play", expanded=True):
    st.latex(
        r"\text{EPA Allowed / Play} = "
        r"\frac{\sum EPA_{\text{opponent plays}}}{\text{Opponent plays}}"
    )
    st.caption("Lower is better.")

with st.expander("Pass EPA Allowed / Play"):
    st.latex(
        r"\text{Pass EPA Allowed / Play} = "
        r"\frac{\sum EPA_{\text{opponent pass plays}}}{\text{Opponent pass plays}}"
    )

with st.expander("Rush EPA Allowed / Play"):
    st.latex(
        r"\text{Rush EPA Allowed / Play} = "
        r"\frac{\sum EPA_{\text{opponent rush plays}}}{\text{Opponent rush plays}}"
    )

with st.expander("Success Rate Allowed"):
    st.latex(
        r"\text{Success Rate Allowed} = "
        r"\frac{\#(\text{Opponent plays with } EPA > 0)}{\text{Opponent plays}}"
    )
    st.caption("Lower is better.")

with st.expander("Sack Rate"):
    st.latex(
        r"\text{Sack Rate} = "
        r"\frac{\text{Sacks}}{\text{Opponent pass plays}}"
    )
    st.caption("Higher is better.")

with st.expander("Turnovers Forced Rate"):
    st.latex(
        r"\text{Turnovers Forced Rate} = "
        r"\frac{\text{Opponent turnover plays}}{\text{Opponent plays}}"
    )
    st.caption("Higher is better.")

with st.expander("Points Allowed / Game"):
    st.latex(
        r"\text{Points Allowed / Game} = "
        r"\frac{\sum \text{Opponent points}}{\text{Games}}"
    )
    st.caption("Lower is better.")

st.header("4. Building a Score Prediction")

st.markdown("""
Each team's score prediction uses **16 inputs**:

- 1 home/away indicator
- 8 offensive features for that team
- 7 defensive features for its opponent
""")

st.code("""[
    is_home,

    off_epa_per_play,
    off_pass_epa_per_play,
    off_rush_epa_per_play,
    off_success_rate,
    off_pass_rate,
    off_turnover_rate,
    off_points_per_game,
    off_plays_per_game,

    def_epa_allowed_per_play,
    def_pass_epa_allowed_per_play,
    def_rush_epa_allowed_per_play,
    def_success_rate_allowed,
    def_sack_rate,
    def_turnover_forced_rate,
    def_points_allowed_per_game,
]""", language="python")

st.latex(
    r"\widehat{\text{Points}} = "
    r"f(\text{Team Offense},\text{Opponent Defense},\text{Home/Away})"
)

st.header("5. Random Forest Regression")

st.markdown("""
The function above is learned with a **Random Forest Regressor**. Each decision
tree makes its own score prediction, and the forest averages them.
""")

st.latex(
    r"\hat{y} = \frac{1}{T}\sum_{t=1}^{T}\hat{y}_t"
)

st.markdown("""
where $T$ is the number of trees and each tree contributes one score prediction.
""")

st.header("6. Model Validation")

mae_tab, rmse_tab, r2_tab = st.tabs(["MAE", "RMSE", "R²"])

with mae_tab:
    st.latex(
        r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|"
    )
    st.write("Average absolute scoring error in points.")

with rmse_tab:
    st.latex(
        r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}"
    )
    st.write("Like MAE, but large misses receive a stronger penalty.")

with r2_tab:
    st.latex(
        r"R^2 = 1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}"
    )
    st.write("Measures how much of the variation in scoring is explained by the model.")

st.header("7. From Predicted Scores to Probabilities")

st.markdown("""
The model first produces one expected score. To represent uncertainty, StockPig
uses the errors from chronologically held-out validation games.
""")

st.latex(
    r"\text{Residual} = \text{Actual Points} - \text{Predicted Points}"
)

st.markdown("Each Monte Carlo simulation samples a historical validation residual:")

st.latex(
    r"\text{Simulated Score} = \text{Predicted Score} + \text{Sampled Residual}"
)

st.markdown("""
Repeating this thousands of times creates a score distribution for both teams.
""")

st.latex(
    r"P(\text{Home Win}) = "
    r"\frac{\#(\text{Home Score}>\text{Away Score})}{\text{Simulations}}"
)

st.latex(
    r"P(\text{Score}=s) = "
    r"\frac{\#(\text{Simulations with score }s)}{\text{Simulations}}"
)

st.header("8. SEC Ranking Math")

st.markdown("""
The SEC rankings do **not** use the Random Forest. Each feature is converted to
a percentile relative to the other SEC teams.

For higher-is-better statistics, the best team receives the highest percentile.
For lower-is-better statistics, the ranking direction is reversed.
""")

st.latex(
    r"\text{Feature Score} = 100 \times \text{SEC Percentile Rank}"
)

st.subheader("Offense Score")

st.latex(
    r"\text{Offense Score} = "
    r"\frac{\sum_{j=1}^{6}\text{Offensive Feature Percentile}_j}{6}"
)

st.markdown("""
Included offensive quality metrics:

1. EPA / play
2. Pass EPA / play
3. Rush EPA / play
4. Success rate
5. Turnover rate — lower is better
6. Points / game

Pass rate and plays/game are displayed but do not affect the composite.
""")

st.subheader("Defense Score")

st.latex(
    r"\text{Defense Score} = "
    r"\frac{\sum_{j=1}^{7}\text{Defensive Feature Percentile}_j}{7}"
)

st.markdown("""
Included defensive metrics:

1. EPA allowed / play — lower is better
2. Pass EPA allowed / play — lower is better
3. Rush EPA allowed / play — lower is better
4. Success rate allowed — lower is better
5. Sack rate — higher is better
6. Turnovers forced rate — higher is better
7. Points allowed / game — lower is better
""")

st.subheader("Overall SEC Score")

st.latex(
    r"\text{Overall Score} = "
    r"\frac{\text{Offense Score}+\text{Defense Score}}{2}"
)

st.markdown("""
The 16 SEC teams are sorted by Overall Score. This is a **relative conference
ranking**, not an absolute football rating.
""")

st.header("9. Preventing Data Leakage")

st.markdown("""
If the app is evaluating teams entering Week 6, only earlier information may be
used:
""")

st.code("""Used:
Weeks 1-5
Previous-season games

Not used:
Week 6
Weeks 7+
""", language="text")

st.markdown("""
This rule applies to both the score predictor and the SEC rankings.
""")

st.divider()

st.caption(
    "StockPig's outputs are probabilistic analytical estimates. A predicted "
    "score is the center of a distribution of possible outcomes, not a claim "
    "that an exact final score will occur."
)
