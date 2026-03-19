import streamlit as st
from streamlit_config import init_app
import pandas as pd
import plotly.express as px
import numpy as np

from utils import data_loader, features
from models import PoissonModel, RFModel, XGBModel, NNModel, EnsembleModel

init_app()

st.title("⚽ EPL Match Predictor")
st.markdown("**Production ML Dashboard for Premier League Score Predictions**")


@st.cache_data
def load_all_data():
    df = data_loader.load_epl_data()
    team_list = data_loader.get_team_list(df)
    X_all, y_all = features.create_features(df)
    return df, team_list, X_all


df, team_list, X_all = load_all_data()


@st.cache_resource
def load_models():
    poisson = PoissonModel.load("models/trained_models/poisson.pkl")

    rf = RFModel.load(
        "models/trained_models/rf_home.pkl",
        "models/trained_models/rf_away.pkl"
    )
    xgb = XGBModel.load(
        "models/trained_models/xgb_home.pkl",
        "models/trained_models/xgb_away.pkl"
    )

    nn = NNModel.load(
        "models/trained_models/nn_home.pkl",
        "models/trained_models/nn_away.pkl",
        "models/trained_models/nn_scaler_home.pkl",
        "models/trained_models/nn_scaler_away.pkl"
    )

    ensemble = EnsembleModel()
    ensemble.load_all()

    return {
        "poisson": poisson,
        "rf": rf,
        "xgb": xgb,
        "nn": nn,
        "ensemble": ensemble,
    }


models = load_models()


def safe_last_rolling_mean(series, window=5):
    if series.empty:
        return None
    rolled = series.rolling(window, min_periods=1).mean()
    if rolled.empty:
        return None
    value = rolled.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def get_last_n_home_matches(team: str, data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    team_home = data[data["HomeTeam"] == team].sort_values("Date")
    return team_home.tail(n)


def get_last_n_away_matches(team: str, data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    team_away = data[data["AwayTeam"] == team].sort_values("Date")
    return team_away.tail(n)


def get_last_n_matches(team: str, data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    team_matches = data[
        (data["HomeTeam"] == team) | (data["AwayTeam"] == team)
    ].sort_values("Date")
    return team_matches.tail(n)


def compute_team_points(match_row, team):
    if match_row["HomeTeam"] == team:
        if match_row["FTR"] == "H":
            return 3
        if match_row["FTR"] == "D":
            return 1
        return 0
    else:
        if match_row["FTR"] == "A":
            return 3
        if match_row["FTR"] == "D":
            return 1
        return 0


def build_match_features(home_team: str, away_team: str, selected_season: str) -> dict:
    """
    Build one feature row for a future match using historical data
    up to and including the selected season.
    """
    historical_df = df[df["Season"] <= selected_season].copy()
    feature_dict = {}

    for n in [3, 5]:
        home_home = get_last_n_home_matches(home_team, historical_df, n)
        away_away = get_last_n_away_matches(away_team, historical_df, n)

        feature_dict[f"home_gf_h{n}"] = float(home_home["FTHG"].mean()) if not home_home.empty else 0.0
        feature_dict[f"home_ga_h{n}"] = float(home_home["FTAG"].mean()) if not home_home.empty else 0.0
        feature_dict[f"away_gf_a{n}"] = float(away_away["FTAG"].mean()) if not away_away.empty else 0.0
        feature_dict[f"away_ga_a{n}"] = float(away_away["FTHG"].mean()) if not away_away.empty else 0.0

    home_matches = get_last_n_home_matches(home_team, historical_df, 5)
    away_matches = get_last_n_away_matches(away_team, historical_df, 5)

    if not home_matches.empty:
        home_points = home_matches["FTR"].map({"H": 3, "D": 1, "A": 0})
        feature_dict["home_form"] = float(home_points.mean())
    else:
        feature_dict["home_form"] = 0.0

    if not away_matches.empty:
        away_points = away_matches["FTR"].map({"A": 3, "D": 1, "H": 0})
        feature_dict["away_form"] = float(away_points.mean())
    else:
        feature_dict["away_form"] = 0.0

    h2h = historical_df[
        (
            (historical_df["HomeTeam"] == home_team) & (historical_df["AwayTeam"] == away_team)
        ) |
        (
            (historical_df["HomeTeam"] == away_team) & (historical_df["AwayTeam"] == home_team)
        )
    ].sort_values("Date").tail(10)

    if not h2h.empty:
        home_for = []
        home_against = []
        away_for = []
        away_against = []

        for _, row in h2h.iterrows():
            if row["HomeTeam"] == home_team:
                home_for.append(row["FTHG"])
                home_against.append(row["FTAG"])
            else:
                home_for.append(row["FTAG"])
                home_against.append(row["FTHG"])

            if row["HomeTeam"] == away_team:
                away_for.append(row["FTHG"])
                away_against.append(row["FTAG"])
            else:
                away_for.append(row["FTAG"])
                away_against.append(row["FTHG"])

        feature_dict["h2h_home_adv"] = float(np.mean(home_for) - np.mean(home_against))
        feature_dict["h2h_away_adv"] = float(np.mean(away_for) - np.mean(away_against))
    else:
        feature_dict["h2h_home_adv"] = 0.0
        feature_dict["h2h_away_adv"] = 0.0

    if "home_shots" in X_all.columns:
        hs = get_last_n_home_matches(home_team, historical_df, 5)
        feature_dict["home_shots"] = float(hs["HS"].mean()) if ("HS" in hs.columns and not hs.empty) else 0.0

    if "away_shots" in X_all.columns:
        aws = get_last_n_away_matches(away_team, historical_df, 5)
        feature_dict["away_shots"] = float(aws["AS"].mean()) if ("AS" in aws.columns and not aws.empty) else 0.0

    if "home_shots_on_target" in X_all.columns:
        hsot = get_last_n_home_matches(home_team, historical_df, 5)
        feature_dict["home_shots_on_target"] = float(hsot["HST"].mean()) if ("HST" in hsot.columns and not hsot.empty) else 0.0

    if "away_shots_on_target" in X_all.columns:
        awsot = get_last_n_away_matches(away_team, historical_df, 5)
        feature_dict["away_shots_on_target"] = float(awsot["AST"].mean()) if ("AST" in awsot.columns and not awsot.empty) else 0.0

    feature_dict["home_adv"] = 0.3
    feature_dict["season_progress"] = 0.5

    home_recent = get_last_n_matches(home_team, historical_df, 5)
    away_recent = get_last_n_matches(away_team, historical_df, 5)

    if not home_recent.empty:
        home_strength_points = [compute_team_points(row, home_team) for _, row in home_recent.iterrows()]
        feature_dict["home_strength"] = float(np.mean(home_strength_points))
    else:
        feature_dict["home_strength"] = 1.0

    if not away_recent.empty:
        away_strength_points = [compute_team_points(row, away_team) for _, row in away_recent.iterrows()]
        feature_dict["away_strength"] = float(np.mean(away_strength_points))
    else:
        feature_dict["away_strength"] = 1.0

    for col in X_all.columns:
        if col not in feature_dict:
            feature_dict[col] = 0.0

    return feature_dict


page = st.sidebar.selectbox(
    "Choose page",
    [
        "Match Predictor",
        "Model Comparison",
        "Team Statistics",
        "Historical Results",
    ],
)

if page == "Match Predictor":
    st.header("🔮 Match Prediction")

    season_values = sorted(df["Season"].dropna().unique()) if "Season" in df.columns else []
    selected_season = st.selectbox(
        "Season",
        season_values,
        index=len(season_values) - 1 if season_values else 0
    )

    season_df = df[df["Season"] == selected_season].copy() if season_values else df.copy()
    season_team_list = sorted(
        set(season_df["HomeTeam"].dropna().unique()) |
        set(season_df["AwayTeam"].dropna().unique())
    )

    st.caption(f"Using data up to season: {selected_season}")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", season_team_list)
    with col2:
        away_team = st.selectbox(
            "Away Team",
            season_team_list,
            index=1 if len(season_team_list) > 1 else 0
        )

    if home_team == away_team:
        st.warning("Please select different teams!")
        st.stop()

    if st.button("Predict Match ⚽", type="primary"):
        with st.spinner("Generating predictions..."):
            pois_pred = models["poisson"].predict_proba(home_team, away_team)

            match_features = build_match_features(home_team, away_team, selected_season)
            X_match = pd.DataFrame([match_features]).reindex(columns=X_all.columns, fill_value=0.0)

            rf_pred = models["rf"].predict_match(match_features)

            xgb_home_pred, xgb_away_pred = models["xgb"].predict(X_match)
            xgb_home = float(xgb_home_pred[0])
            xgb_away = float(xgb_away_pred[0])

            nn_home_pred, nn_away_pred = models["nn"].predict(X_match)
            nn_home = float(nn_home_pred[0])
            nn_away = float(nn_away_pred[0])

            ensemble_home = np.mean(
                [pois_pred["home_goals"], rf_pred["home_goals"], xgb_home, nn_home]
            )
            ensemble_away = np.mean(
                [pois_pred["away_goals"], rf_pred["away_goals"], xgb_away, nn_away]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Home Goals",
                    f"{ensemble_home:.1f}",
                    delta=f"Poisson: {pois_pred['home_goals']:.1f}",
                )
            with col2:
                st.metric(
                    "Away Goals",
                    f"{ensemble_away:.1f}",
                    delta=f"Poisson: {pois_pred['away_goals']:.1f}",
                )

            st.markdown("### Predicted Score")
            st.success(
                f"{home_team} {int(round(ensemble_home))} - {int(round(ensemble_away))} {away_team}"
            )

            probs_df = pd.DataFrame(
                {
                    "Outcome": ["Home Win", "Draw", "Away Win"],
                    "Probability": [
                        pois_pred["home_win"],
                        pois_pred["draw"],
                        pois_pred["away_win"],
                    ],
                }
            )

            fig_prob = px.bar(
                probs_df,
                x="Outcome",
                y="Probability",
                title="Match Outcome Probabilities",
                color="Outcome",
                color_discrete_map={
                    "Home Win": "#EF0107",
                    "Draw": "#FFD700",
                    "Away Win": "#024FA3",
                },
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            max_g = 6
            score_mat = pois_pred["score_matrix"]
            fig_heat = px.imshow(
                score_mat[: max_g + 1, : max_g + 1],
                labels={
                    "x": "Away Goals",
                    "y": "Home Goals",
                    "color": "Probability",
                },
                title="Most Likely Scorelines (Poisson)",
                text_auto=".2f",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            with st.expander("Show model input features"):
                st.dataframe(X_match, use_container_width=True)

elif page == "Model Comparison":
    st.header("📊 Model Comparison")

    season_values = sorted(df["Season"].dropna().unique()) if "Season" in df.columns else []
    selected_season = st.selectbox(
        "Season",
        season_values,
        index=len(season_values) - 1 if season_values else 0
    )

    season_df = df[df["Season"] == selected_season].copy()

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", team_list, key="mc_home")
    with col2:
        away_team = st.selectbox("Away Team", team_list, key="mc_away")

    if home_team == away_team:
        st.warning("Please select different teams!")
        st.stop()

    if st.button("Compare Models ⚽"):
        with st.spinner("Comparing models..."):

            match_features = build_match_features(home_team, away_team, selected_season)
            X_match = pd.DataFrame([match_features]).reindex(columns=X_all.columns, fill_value=0.0)

            pois = models["poisson"].predict_proba(home_team, away_team)

            rf = models["rf"].predict_match(match_features)

            xgb_home_pred, xgb_away_pred = models["xgb"].predict(X_match)
            nn_home_pred, nn_away_pred = models["nn"].predict(X_match)

            results = pd.DataFrame({
                "Model": ["Poisson", "Random Forest", "XGBoost", "Neural Network"],
                "Home Goals": [
                    pois["home_goals"],
                    rf["home_goals"],
                    float(xgb_home_pred[0]),
                    float(nn_home_pred[0])
                ],
                "Away Goals": [
                    pois["away_goals"],
                    rf["away_goals"],
                    float(xgb_away_pred[0]),
                    float(nn_away_pred[0])
                ]
            })

            st.subheader("📊 Predictions Comparison")
            st.dataframe(results, use_container_width=True)

            # Visualization
            fig = px.bar(
                results,
                x="Model",
                y=["Home Goals", "Away Goals"],
                barmode="group",
                title="Model Predictions Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Team Statistics":
    st.header("📈 Team Statistics")

    selected_team = st.selectbox("Select Team", team_list)

    team_home = df[df["HomeTeam"] == selected_team].copy()
    team_away = df[df["AwayTeam"] == selected_team].copy()

    home_points = team_home["FTR"].map({"H": 3, "D": 1, "A": 0})
    away_points = team_away["FTR"].map({"A": 3, "D": 1, "H": 0})

    home_form = safe_last_rolling_mean(home_points, window=5)
    away_form = safe_last_rolling_mean(away_points, window=5)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Home Form (PPG)",
            f"{home_form:.2f}" if home_form is not None else "N/A"
        )
    with col2:
        st.metric(
            "Away Form (PPG)",
            f"{away_form:.2f}" if away_form is not None else "N/A"
        )

    st.subheader("Recent Matches")
    recent_matches = df[
        (df["HomeTeam"] == selected_team) | (df["AwayTeam"] == selected_team)
    ].sort_values("Date").tail(10)

    display_cols = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"] if c in recent_matches.columns]
    st.dataframe(recent_matches[display_cols], use_container_width=True)

elif page == "Historical Results":
    st.header("📋 Historical Results Explorer")

    season_values = sorted(df["Season"].dropna().unique()) if "Season" in df.columns else []
    default_seasons = season_values[-3:] if len(season_values) >= 3 else season_values

    season_filter = st.multiselect(
        "Seasons",
        season_values,
        default=default_seasons,
    )
    team_filter = st.text_input("Team filter")

    filtered_df = df.copy()

    if season_values and season_filter:
        filtered_df = filtered_df[filtered_df["Season"].isin(season_filter)].copy()

    if team_filter:
        filtered_df = filtered_df[
            filtered_df["HomeTeam"].str.contains(team_filter, case=False, na=False)
            | filtered_df["AwayTeam"].str.contains(team_filter, case=False, na=False)
        ]

    result_cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    available_cols = [col for col in result_cols if col in filtered_df.columns]

    st.dataframe(filtered_df[available_cols].tail(100), use_container_width=True)

st.markdown("---")
st.markdown(
    "**Disclaimer**: Predictions are probabilistic. Football outcomes vary! "
    "Data from football-data.co.uk"
)

