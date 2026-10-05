StockPig opponent score-distribution overlay patch

Apply from the StockPig repository root:

    unzip -o stockpig_opponent_distribution_patch.zip -d .
    python apply_opponent_distribution_patch.py

Then run:

    uv run streamlit run app.py

Changes:
- Each recommended-pick score graph now includes BOTH teams.
- Model-projected winner = blue.
- Model-projected loser = red.
- Both distributions share the same score/probability axes.
- Streamlit uses side-by-side bars at each score.
- The downloadable HTML report uses matching blue/red side-by-side SVG bars.
- Both teams receive expected score, win probability, and 80% simulated range.
- Double-pick weeks still show exactly two matchup graphs.

Backup created:
    pages/survival_planner.py.pre_opponent_distribution_backup
