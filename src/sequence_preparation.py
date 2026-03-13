import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

def prepare_sequences(file, time_steps=10):

    df = pd.read_csv(file)

    # clean column names
    df.columns = df.columns.str.strip()

    # remove invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # dataset balancing
    benign = df[df["Label"] == "BENIGN"]
    attack = df[df["Label"] != "BENIGN"]

    benign_downsampled = resample(
        benign,
        replace=False,
        n_samples=len(attack),
        random_state=42
    )

    df = pd.concat([benign_downsampled, attack])

    # separate features and label
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # remove highly correlated features
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X = X.drop(columns=to_drop)

    # encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    sequences = []
    labels = []

    for i in range(len(X) - time_steps):

        sequences.append(X[i:i+time_steps])
        labels.append(y[i+time_steps])

    sequences = np.array(sequences)
    labels = np.array(labels)

    # reshape for ConvLSTM
    sequences = sequences.reshape(
        sequences.shape[0],
        time_steps,
        1,
        sequences.shape[2],
        1
    )

    return sequences, labels, scaler, encoder