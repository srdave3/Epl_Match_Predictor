import pandas as pd
from pathlib import Path


def load_epl_data() -> pd.DataFrame:
    """Load and clean EPL historical data."""
    data_path = Path("data/epl_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"EPL data not found at {data_path}. "
            "Run `python data/download_data.py` first."
        )

    df = pd.read_csv(data_path)

    # Parse dates safely
    if "Date" not in df.columns:
        raise ValueError("Column 'Date' not found in dataset.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Ensure required columns exist
    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Clean rows
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0)].copy()

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # Optional team-name cleanup
    team_mapping = {
        "Arsenal": "Arsenal",
        "Aston Villa": "Aston Villa",
        "Burnley": "Burnley",
        "Brighton": "Brighton",
        "Bournemouth": "Bournemouth",
    }

    df["HomeTeam"] = df["HomeTeam"].map(team_mapping).fillna(df["HomeTeam"])
    df["AwayTeam"] = df["AwayTeam"].map(team_mapping).fillna(df["AwayTeam"])

    if "Season" in df.columns and not df["Season"].dropna().empty:
        print(f"Loaded {len(df)} matches from {df['Season'].min()} to {df['Season'].max()}")
    else:
        print(f"Loaded {len(df)} matches")

    return df


def get_team_list(df: pd.DataFrame) -> list:
    """Return sorted unique team names from HomeTeam and AwayTeam."""
    if df.empty:
        return []

    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise ValueError("DataFrame must contain 'HomeTeam' and 'AwayTeam' columns.")

    teams = sorted(set(df["HomeTeam"].dropna().unique()) | set(df["AwayTeam"].dropna().unique()))
    return teams


if __name__ == "__main__":
    df = load_epl_data()
    print(df.head())
    print("Teams:", get_team_list(df))
