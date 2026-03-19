import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from utils.features import prepare_training_data
from typing import Tuple

class NNModel:
    def __init__(self):
        self.home_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.away_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.scaler_home = StandardScaler()
        self.scaler_away = StandardScaler()
        self.feature_names = None
    
    def fit(self):
        (X_train, y_home_train, y_away_train), _ = prepare_training_data()
        
        self.feature_names = X_train.columns.tolist()
        
        X_scaled_home = self.scaler_home.fit_transform(X_train)
        X_scaled_away = self.scaler_away.fit_transform(X_train)
        
        print('Training NN home goals...')
        self.home_model.fit(X_scaled_home, y_home_train)
        
        print('Training NN away goals...')
        self.away_model.fit(X_scaled_away, y_away_train)
        
        print('NN models trained!')
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled_home = self.scaler_home.transform(X)
        X_scaled_away = self.scaler_away.transform(X)
        home_pred = self.home_model.predict(X_scaled_home)
        away_pred = self.away_model.predict(X_scaled_away)
        return home_pred, away_pred
    
    def save(self, path_home: str, path_away: str, path_scaler_h: str, path_scaler_a: str):
        joblib.dump(self.home_model, path_home)
        joblib.dump(self.away_model, path_away)
        joblib.dump(self.scaler_home, path_scaler_h)
        joblib.dump(self.scaler_away, path_scaler_a)
    
    @classmethod
    def load(cls, path_home: str, path_away: str, path_scaler_h: str, path_scaler_a: str):
        model = cls()
        model.home_model = joblib.load(path_home)
        model.away_model = joblib.load(path_away)
        model.scaler_home = joblib.load(path_scaler_h)
        model.scaler_away = joblib.load(path_scaler_a)
        return model

if __name__ == '__main__':
    model = NNModel()
    model.fit()

