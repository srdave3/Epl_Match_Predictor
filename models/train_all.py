import time
from pathlib import Path

from models.poisson_model import PoissonModel
from models.rf_model import RFModel
from models.xgb_model import XGBModel
from models.nn_model import NNModel
from utils.data_loader import load_epl_data


def main():
    models_dir = Path("models/trained_models")
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_epl_data()

    print(f"Loaded dataset with {len(df)} rows")
    print("")

    # 1. Poisson
    print("1. Training Poisson Model...")
    start = time.time()

    poisson_df = df.sample(frac=0.05, random_state=42).copy()
    print(f"   Using {len(poisson_df)} sampled matches for Poisson training")

    poisson = PoissonModel()
    poisson.fit(poisson_df)
    poisson.save(str(models_dir / "poisson.pkl"))

    print(f"   Saved: {models_dir / 'poisson.pkl'}")
    print(f"   Done in {time.time() - start:.2f} seconds")
    print("")

    # 2. Random Forest
    print("2. Training RF Model...")
    start = time.time()

    rf = RFModel()
    rf.fit()
    rf.save(
        str(models_dir / "rf_home.pkl"),
        str(models_dir / "rf_away.pkl"),
    )

    print(f"   Saved: {models_dir / 'rf_home.pkl'}")
    print(f"   Saved: {models_dir / 'rf_away.pkl'}")
    print(f"   Done in {time.time() - start:.2f} seconds")
    print("")

    # 3. XGBoost
    print("3. Training XGBoost Model...")
    start = time.time()

    xgb = XGBModel()
    xgb.fit()
    xgb.save(
        str(models_dir / "xgb_home.pkl"),
        str(models_dir / "xgb_away.pkl"),
    )

    print(f"   Saved: {models_dir / 'xgb_home.pkl'}")
    print(f"   Saved: {models_dir / 'xgb_away.pkl'}")
    print(f"   Done in {time.time() - start:.2f} seconds")
    print("")

    # 4. Neural Network
    print("4. Training NN Model...")
    start = time.time()

    nn = NNModel()
    nn.fit()
    nn.save(
        str(models_dir / "nn_home.pkl"),
        str(models_dir / "nn_away.pkl"),
        str(models_dir / "nn_scaler_home.pkl"),
        str(models_dir / "nn_scaler_away.pkl"),
    )

    print(f"   Saved: {models_dir / 'nn_home.pkl'}")
    print(f"   Saved: {models_dir / 'nn_away.pkl'}")
    print(f"   Saved: {models_dir / 'nn_scaler_home.pkl'}")
    print(f"   Saved: {models_dir / 'nn_scaler_away.pkl'}")
    print(f"   Done in {time.time() - start:.2f} seconds")
    print("")

    print("✅ All models trained and saved!")
    print(f"Models directory: {models_dir.resolve()}")


if __name__ == "__main__":
    main()
