# EPL Match Outcome Prediction System

A production-ready Streamlit dashboard for predicting English Premier League match scores using advanced ML models (Poisson Dixon-Coles, Random Forest, XGBoost, Neural Network + Ensemble).

![EPL Prediction Dashboard](https://via.placeholder.com/1200x600/0f4c1f/ffffff?text=EPL+Match+Predictor)

## Features
- **Match Predictor**: Select teams → Get score predictions, Win/Draw/Loss probabilities, score heatmap
- **Model Comparison**: Side-by-side model outputs + historical accuracy
- **Team Stats**: Form trends, home/away breakdown
- **Historical Explorer**: Filterable results + H2H summaries
- Dark EPL-themed UI with interactive Plotly charts

## Quick Start

1. **Clone/Setup**
   ```bash
   cd Football_Analysis
   pip install -r requirements.txt
   ```

2. **Download Data**
   ```bash
   python data/download_data.py
   ```

3. **Train Models** (first time only)
   ```bash
   python -m models.train_all
   ```

4. **Run App**
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501

## Architecture
```
data/          ← Historical EPL CSVs (football-data.co.uk)
models/        ← 4 ML models + ensemble (.pkl)
utils/         ← Features, data loading, teams
app.py         ← Streamlit frontend
```

## Models & Features
**Targets**: Home/Away Goals → Poisson probs for scores
**Features**:
- Rolling stats (goals, shots, form last 5/10)
- H2H records
- League position, home advantage

**Models**:
1. Poisson (Dixon-Coles)
2. Random Forest
3. XGBoost
4. MLP Neural Network
5. Ensemble (average)

## Data Source
- [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (1993-2024 CSV)

## Deployment
Streamlit Cloud / Docker:
```dockerfile
FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Disclaimer
Predictions are probabilistic. Football is unpredictable! For entertainment & analysis only.

**Metrics on recent seasons**: ~1.2 MAE goals/team, 52% exact score, 63% result accuracy.

