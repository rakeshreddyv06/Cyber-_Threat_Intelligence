import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sequence_preparation import prepare_sequences


print("Loading dataset and preparing sequences...")

X, y, scaler, encoder = prepare_sequences("data/selected_dataset.csv")

print("Dataset shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Building ConvLSTM model...")

model = Sequential()

model.add(
    ConvLSTM2D(
        filters=64,
        kernel_size=(1,3),
        activation="relu",
        return_sequences=True,
        input_shape=(X.shape[1],1,X.shape[3],1)
    )
)

model.add(
    ConvLSTM2D(
        filters=32,
        kernel_size=(1,3),
        activation="relu"
    )
)

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))

model.add(Dense(len(np.unique(y)), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


print("Training model...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)


print("Evaluating model...")

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)


print("Generating classification report...")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred))


print("Saving model...")

model.save("model/convlstm_model.h5")

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoder, "model/encoder.pkl")

print("Model saved successfully.")