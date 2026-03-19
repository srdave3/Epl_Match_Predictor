import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict
from utils.data_loader import load_epl_data


class PoissonModel:
    def __init__(self):
        self.attack = {}
        self.defense = {}
        self.home_adv = 0.0
        self.rho = 0.0
        self.teams = []
        self.team_to_idx = {}

    def negative_log_likelihood(self, params, data):
        n_teams = len(self.teams)

        attack = np.exp(params[:n_teams])
        defense = np.exp(params[n_teams:2 * n_teams])
        home_adv = params[2 * n_teams]
        rho = np.tanh(params[2 * n_teams + 1])

        ll = 0.0

        for _, row in data.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            goals_h = int(row["FTHG"])
            goals_a = int(row["FTAG"])

            home_idx = self.team_to_idx[home]
            away_idx = self.team_to_idx[away]

            lambda_h = attack[home_idx] * defense[away_idx] * np.exp(home_adv)
            lambda_a = attack[away_idx] * defense[home_idx]

            lambda_h = max(lambda_h, 1e-10)
            lambda_a = max(lambda_a, 1e-10)

            ll_h = poisson.logpmf(goals_h, lambda_h)
            ll_a = poisson.logpmf(goals_a, lambda_a)

            if (goals_h == 0 and goals_a == 0) or (goals_h + goals_a <= 2):
                tau = 1 - (goals_h * goals_a * rho / (lambda_h * lambda_a))
                tau = max(tau, 1e-10)
                ll_h += np.log(tau)
                ll_a += np.log(tau)

            ll += ll_h + ll_a

        return -ll

    def fit(self, data):
        data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()

        self.teams = sorted(set(data["HomeTeam"]).union(set(data["AwayTeam"])))
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        n_teams = len(self.teams)

        init_params = np.zeros(2 * n_teams + 2)
        init_params[2 * n_teams] = 0.2

        result = minimize(
            self.negative_log_likelihood,
            init_params,
            args=(data,),
            method="L-BFGS-B",
            options={"maxiter": 50},
        )

        params = result.x

        self.attack = {team: float(params[i]) for i, team in enumerate(self.teams)}
        self.defense = {team: float(params[n_teams + i]) for i, team in enumerate(self.teams)}
        self.home_adv = float(params[2 * n_teams])
        self.rho = float(np.tanh(params[2 * n_teams + 1]))

        print("Poisson model trained!")
        print(f"Home adv: {np.exp(self.home_adv):.3f}, Rho: {self.rho:.3f}")

    def predict_home_goals(self, home: str, away: str) -> float:
        return float(np.exp(self.attack[home] + self.defense[away] + self.home_adv))

    def predict_away_goals(self, home: str, away: str) -> float:
        return float(np.exp(self.attack[away] + self.defense[home]))

    def predict_proba(self, home: str, away: str, max_goals: int = 6) -> Dict:
        lam_h = self.predict_home_goals(home, away)
        lam_a = self.predict_away_goals(home, away)

        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for gh in range(max_goals + 1):
            for ga in range(max_goals + 1):
                score_matrix[gh, ga] = poisson.pmf(gh, lam_h) * poisson.pmf(ga, lam_a)

        home_win = float(np.sum(np.tril(score_matrix, -1)))
        draw = float(np.sum(np.diag(score_matrix)))
        away_win = float(np.sum(np.triu(score_matrix, 1)))

        return {
            "home_goals": lam_h,
            "away_goals": lam_a,
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "most_likely_score": np.unravel_index(np.argmax(score_matrix), score_matrix.shape),
            "score_matrix": score_matrix,
        }

    def save(self, path: str):
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path: str):
        model = cls()
        model.__dict__.update(joblib.load(path))
        return model


if __name__ == "__main__":
    df = load_epl_data()
    model = PoissonModel()
    model.fit(df.sample(frac=0.1, random_state=42))
    print(model.predict_proba("Arsenal", "Chelsea"))

