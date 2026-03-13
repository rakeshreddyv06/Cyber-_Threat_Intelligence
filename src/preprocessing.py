import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):

    df = pd.read_csv(path)

    # remove column spaces
    df.columns = df.columns.str.strip()

    # replace infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    return df


def preprocess(df):

    X = df.drop("Label", axis=1)

    y = df["Label"]

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, y, scaler, encoder