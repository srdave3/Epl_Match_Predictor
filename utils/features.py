import pandas as pd
import numpy as np
from typing import Tuple
from utils.data_loader import load_epl_data


def _result_points(result: str, team_side: str) -> int:
    """
    Convert full-time result to points for a team.
    team_side: 'home' or 'away'
    """
    if pd.isna(result):
        return 0

    if team_side == 'home':
        if result == 'H':
            return 3
        if result == 'D':
            return 1
        return 0

    if team_side == 'away':
        if result == 'A':
            return 3
        if result == 'D':
            return 1
        return 0

    return 0


def _mean_or_zero(series: pd.Series) -> float:
    """Return series mean or 0.0 if empty/all missing."""
    if series is None or len(series) == 0:
        return 0.0
    value = series.mean()
    return 0.0 if pd.isna(value) else float(value)


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create features and targets for match prediction."""
    if df.empty:
        print("create_features received an empty DataFrame")
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=["home_goals", "away_goals"])
        )

    df = df.sort_values("Date").reset_index(drop=True).copy()

    required_cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    feature_cols = []

    # 1. Rolling team stats (using only previous matches to avoid leakage)
    for n in [3, 5]:
        # Home team home attack/defense form
        df[f"home_gf_h{n}"] = (
            df.groupby("HomeTeam")["FTHG"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        )
        df[f"home_ga_h{n}"] = (
            df.groupby("HomeTeam")["FTAG"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        )

        # Away team away attack/defense form
        df[f"away_gf_a{n}"] = (
            df.groupby("AwayTeam")["FTAG"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        )
        df[f"away_ga_a{n}"] = (
            df.groupby("AwayTeam")["FTHG"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        )

        feature_cols.extend([
            f"home_gf_h{n}", f"home_ga_h{n}",
            f"away_gf_a{n}", f"away_ga_a{n}"
        ])

    # 2. Form points from previous matches only
    df["home_points"] = df["FTR"].map(lambda x: _result_points(x, "home"))
    df["away_points"] = df["FTR"].map(lambda x: _result_points(x, "away"))

    df["home_form"] = (
        df.groupby("HomeTeam")["home_points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df["away_form"] = (
        df.groupby("AwayTeam")["away_points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    feature_cols.extend(["home_form", "away_form"])

    # 3. Head-to-head features using only previous meetings
    df["h2h_home_adv"] = 0.0
    df["h2h_away_adv"] = 0.0

    for idx, row in df.iterrows():
        prior = df.iloc[:idx]

        h2h = prior[
            (
                (prior["HomeTeam"] == row["HomeTeam"]) &
                (prior["AwayTeam"] == row["AwayTeam"])
            ) |
            (
                (prior["HomeTeam"] == row["AwayTeam"]) &
                (prior["AwayTeam"] == row["HomeTeam"])
            )
        ].tail(10)

        if not h2h.empty:
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]

            home_goals_for = []
            home_goals_against = []

            away_goals_for = []
            away_goals_against = []

            for _, past in h2h.iterrows():
                # From current home team's perspective
                if past["HomeTeam"] == home_team:
                    home_goals_for.append(past["FTHG"])
                    home_goals_against.append(past["FTAG"])
                else:
                    home_goals_for.append(past["FTAG"])
                    home_goals_against.append(past["FTHG"])

                # From current away team's perspective
                if past["HomeTeam"] == away_team:
                    away_goals_for.append(past["FTHG"])
                    away_goals_against.append(past["FTAG"])
                else:
                    away_goals_for.append(past["FTAG"])
                    away_goals_against.append(past["FTHG"])

            df.at[idx, "h2h_home_adv"] = np.mean(home_goals_for) - np.mean(home_goals_against)
            df.at[idx, "h2h_away_adv"] = np.mean(away_goals_for) - np.mean(away_goals_against)

    feature_cols.extend(["h2h_home_adv", "h2h_away_adv"])

    # 4. Shots stats if available
    if {"HS", "AS"}.issubset(df.columns):
        df["home_shots"] = (
            df.groupby("HomeTeam")["HS"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        df["away_shots"] = (
            df.groupby("AwayTeam")["AS"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        feature_cols.extend(["home_shots", "away_shots"])

    if {"HST", "AST"}.issubset(df.columns):
        df["home_shots_on_target"] = (
            df.groupby("HomeTeam")["HST"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        df["away_shots_on_target"] = (
            df.groupby("AwayTeam")["AST"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        feature_cols.extend(["home_shots_on_target", "away_shots_on_target"])

    # 5. Base home advantage
    df["home_adv"] = 0.3
    feature_cols.append("home_adv")

    # 6. Season progress
    season_sizes = df.groupby("Season")["Season"].transform("size")
    df["season_progress"] = df.groupby("Season").cumcount() / season_sizes.clip(lower=1)
    feature_cols.append("season_progress")

    # 7. Strength features from previous matches (all matches, not only home-only or away-only)
    df["home_strength"] = 1.0
    df["away_strength"] = 1.0

    for idx, row in df.iterrows():
        current_date = row["Date"]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        prior = df.iloc[:idx]

        home_prior = prior[
            (prior["HomeTeam"] == home_team) | (prior["AwayTeam"] == home_team)
        ]
        away_prior = prior[
            (prior["HomeTeam"] == away_team) | (prior["AwayTeam"] == away_team)
        ]

        if not home_prior.empty:
            home_points = []
            for _, match in home_prior.iterrows():
                if match["HomeTeam"] == home_team:
                    home_points.append(_result_points(match["FTR"], "home"))
                else:
                    home_points.append(_result_points(match["FTR"], "away"))
            df.at[idx, "home_strength"] = float(np.mean(home_points)) if home_points else 1.0

        if not away_prior.empty:
            away_points = []
            for _, match in away_prior.iterrows():
                if match["HomeTeam"] == away_team:
                    away_points.append(_result_points(match["FTR"], "home"))
                else:
                    away_points.append(_result_points(match["FTR"], "away"))
            df.at[idx, "away_strength"] = float(np.mean(away_points)) if away_points else 1.0

    feature_cols.extend(["home_strength", "away_strength"])

    # Final feature matrix
    X = df[feature_cols].fillna(0.0).copy()

    # Targets
    y = pd.DataFrame({
        "home_goals": df["FTHG"].astype(float),
        "away_goals": df["FTAG"].astype(float)
    })

    print(f"Created {len(feature_cols)} features for {len(X)} matches")
    print(f"Feature columns: {feature_cols}")

    return X, y


def prepare_training_data(test_season: str = "2023/2024") -> Tuple:
    """Split data into train and test using a season boundary."""
    df = load_epl_data()

    if "Season" not in df.columns:
        raise ValueError("Column 'Season' not found in dataset.")

    seasons = sorted(df["Season"].dropna().unique())
    if test_season not in seasons:
        raise ValueError(
            f"Season '{test_season}' not found. "
            f"Available seasons include: {seasons[:5]} ... {seasons[-5:]}"
        )

    season_rows = df[df["Season"] == test_season].copy()
    if season_rows.empty:
        raise ValueError(f"No rows found for season '{test_season}'.")

    cutoff_date = season_rows["Date"].min()
    if pd.isna(cutoff_date):
        raise ValueError(f"Cutoff date for season '{test_season}' is invalid.")

    train_df = df[df["Date"] < cutoff_date].copy()
    test_df = df[df["Date"] >= cutoff_date].copy()

    print(f"Cutoff date: {cutoff_date}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    if train_df.empty:
        raise ValueError("Training set is empty after applying cutoff date.")
    if test_df.empty:
        raise ValueError("Test set is empty after applying cutoff date.")

    X_train, y_train = create_features(train_df)
    X_test, y_test = create_features(test_df)

    return (
        X_train,
        y_train["home_goals"],
        y_train["away_goals"]
    ), (
        X_test,
        y_test["home_goals"],
        y_test["away_goals"]
    )


if __name__ == "__main__":
    df = load_epl_data()
    X, y = create_features(df)
    print("Features:", X.columns.tolist())
    print(X.head())
    print(y.head())

