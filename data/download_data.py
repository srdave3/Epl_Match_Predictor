import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session():
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )
    return session


def parse_date_column(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce", format="%d/%m/%Y")


def download_epl_data(start_year=1998, end_year=2024, pause_seconds=1):
    base_url = "https://www.football-data.co.uk/mmz4281/"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    all_data = []
    session = build_session()

    for year in range(start_year, end_year + 1):
        season_code = f"{year % 100:02d}{(year + 1) % 100:02d}"
        season_label = f"{year}/{year + 1}"
        url = f"{base_url}{season_code}/E0.csv"

        print(f"Downloading {season_label}...")

        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()

            try:
                df = pd.read_csv(StringIO(response.text))
            except pd.errors.ParserError:
                df = pd.read_csv(StringIO(response.text), engine="python", on_bad_lines="skip")

            df["Season"] = season_label

            if "Date" in df.columns:
                df["Date"] = parse_date_column(df["Date"])

            columns = [
                "Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF",
                "HY", "AY", "HR", "AR"
            ]
            available_cols = [col for col in columns if col in df.columns]
            df = df[available_cols]

            if "FTHG" in df.columns and "FTAG" in df.columns:
                df = df.dropna(subset=["FTHG", "FTAG"])

            all_data.append(df)
            print(f"✅ {season_label}: {len(df)} matches")
            time.sleep(pause_seconds)

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed {season_label}: network error -> {e}")
        except Exception as e:
            print(f"❌ Failed {season_label}: {e}")

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)

        if "Date" in full_df.columns:
            full_df = full_df.sort_values("Date", na_position="last")

        output_path = data_dir / "epl_data.csv"
        full_df.to_csv(output_path, index=False)

        print(f"\n🎉 Saved {len(full_df)} matches to {output_path}")
        print(f"Seasons: {full_df['Season'].min()} to {full_df['Season'].max()}")
        print(f"Teams: {len(full_df['HomeTeam'].dropna().unique())} unique")
    else:
        print("\nNo data downloaded!")


if __name__ == "__main__":
    download_epl_data()