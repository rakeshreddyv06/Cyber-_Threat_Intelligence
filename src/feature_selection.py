import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def select_important_features(file, top_k=30):

    print("Loading dataset...")

    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    # clean dataset
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print("Dataset shape:", df.shape)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print("Training Random Forest for feature importance...")

    rf = RandomForestClassifier(n_estimators=100)

    rf.fit(X, y)

    importance = rf.feature_importances_

    feature_scores = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })

    feature_scores = feature_scores.sort_values(
        by="importance",
        ascending=False
    )

    selected_features = feature_scores.head(top_k)["feature"]

    print("\nTop Selected Features:")
    print(selected_features)

    # create new dataset
    X_selected = X[selected_features]

    X_selected["Label"] = df["Label"]

    output_file = "data/selected_dataset.csv"

    X_selected.to_csv(output_file, index=False)

    print("\nNew dataset saved at:", output_file)


# RUN FEATURE SELECTION
if __name__ == "__main__":

    select_important_features("data/clean_dataset.csv", top_k=30)