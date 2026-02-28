from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def estimate_anemia_risk(dataframe, user_input: dict):
    required_columns = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV", "Result"]
    if dataframe is None or not all(column in dataframe.columns for column in required_columns):
        raise ValueError("Dataset does not contain the required columns for prediction.")

    working_df = dataframe[required_columns].copy()
    for column in required_columns:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
    working_df = working_df.dropna()

    feature_columns = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
    features = working_df[feature_columns]
    result_series = working_df["Result"]

    ranges = (features.max() - features.min()).replace(0, 1)
    input_row = pd.Series(user_input, index=feature_columns)

    normalized_diff = (features - input_row).abs() / ranges
    distance = (normalized_diff.pow(2).sum(axis=1)).pow(0.5)

    neighbor_count = max(5, min(25, len(working_df)))
    nearest_indices = distance.nsmallest(neighbor_count).index
    nearest_results = result_series.loc[nearest_indices]

    risk_probability = float(nearest_results.mean())
    predicted_label = 1 if risk_probability >= 0.5 else 0
    return risk_probability * 100, predicted_label, neighbor_count
