import pandas as pd


def detect_latest_complete_round(games_meta: pd.DataFrame, round_col: str = "Round") -> int:
    """
    Derive the latest fully-played round from game metadata.

    A round counts as complete once every game in it has a recorded
    home/away score. Returns 0 if no round is complete yet.
    """
    df = games_meta.copy()
    df["_complete"] = df["homescore"].notna() & df["awayscore"].notna()

    complete_by_round = df.groupby(round_col)["_complete"].all()
    completed = complete_by_round[complete_by_round].index

    return int(completed.max()) if len(completed) else 0
