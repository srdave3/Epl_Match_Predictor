def prepare_training_data(test_season: str = '2023/2024') -> Tuple:
    """Split data for training/validation."""
    df = load_epl_data()

    season_rows = df[df['Season'] == test_season]
    if season_rows.empty:
        raise ValueError(
            f"Season '{test_season}' not found. "
            f"Available examples: {sorted(df['Season'].dropna().unique())[:5]} ... "
            f"{sorted(df['Season'].dropna().unique())[-5:]}"
        )

    cutoff_date = season_rows['Date'].min()

    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()

    print(f"Cutoff date: {cutoff_date}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    X_train, y_train = create_features(train_df)
    X_test, y_test = create_features(test_df)

    return (
        X_train,
        y_train['home_goals'],
        y_train['away_goals']
    ), (
        X_test,
        y_test['home_goals'],
        y_test['away_goals']
    )