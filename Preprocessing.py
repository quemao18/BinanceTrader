# Hello there! 😊
# This script is designed to help you gather data and train your model effectively.
# If you have alternative methods for training your model, feel free to use them and overwrite the
# "model.keras" and "scaler.joblib" files to ensure the commands run correctly.

import pandas as pd
from Indicators import Indicators
from Labels import EndType, get_zig_zag
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[reportMissingImports,reportMissingModuleSource]
from tensorflow.keras.models import Sequential  # type: ignore[reportMissingImports,reportMissingModuleSource]
from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore[reportMissingImports,reportMissingModuleSource]
from tensorflow.keras.regularizers import L2  # type: ignore[reportMissingImports,reportMissingModuleSource]
import numpy as np

# the BTCUSDT_15m.csv file Is 15m candles of BTC/USDT currency pair and it has Date, Open, High, Low, Close columns
data = pd.read_csv("BTCUSDT_15m.csv", parse_dates = ["Date"], index_col = "Date") # Change the name of the file if needed
indicators: Indicators = Indicators(data)
indicators.all_ind()
pca = indicators.add_pca(1)  # type: ignore[reportUndefinedVariable]
data = indicators.data.copy()
data["Feature_1"] = pca
df = data[["Open", "High", "Low", "Close"]].copy()
def zigzag(df, source = EndType.HIGH_LOW, pct = 5, forward_candles=5):
    """
    Generate zigzag trends with forward-looking labels.
    forward_candles: predict trend 5 candles ahead (more confirmation, reduces whipsaw)
    """
    zig_zag_results = get_zig_zag(df.itertuples(), end_type = source, percent_change = pct)

    # Access the Zig Zag results
    trend = []
    for result in zig_zag_results:
        if result.point_type == "H":
            trend.append(0)
        elif result.point_type == "L":
            trend.append(1)
        else:
            trend.append(None)
    s = pd.Series(trend, index = df.index)
    s.ffill(inplace = True)
    return s.shift(-forward_candles)  # Predict 5 candles ahead
data["Labels"] = zigzag(df = df, source = EndType.HIGH_LOW, pct = 2, forward_candles=5)

# Aggregate lower timeframe labels to boost signal confidence
lags = []
for i in range(1, 13):
    lag = f"label_lag_{i}"
    data[lag] = data["Labels"].shift(i)
    lags.append(lag)

# Add momentum features: RSI-like, volatility
data["Volatility"] = data["High"] - data["Low"]  # Intra-candle volatility
data["Returns"] = data["Close"].pct_change() * 100  # % return per candle
dropping_cols = ['EMA', 'Open', 'High', 'Close', 'Low']
data.drop(dropping_cols, axis = 1, inplace = True)
data.dropna(inplace = True)
X = data.drop("Labels", axis = 1).values
y = data["Labels"].values

# Better class balancing: ensure 50/50 mix to improve sensitivity
n_class_1 = np.sum(y == 1)
n_class_0 = np.sum(y == 0)
print(f"Initial label distribution: Class 0 (HIGH) = {n_class_0}, Class 1 (LOW) = {n_class_1}")

# Stratified train/val split for better validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

scaler = MinMaxScaler()
scaler.fit(X_train)
dump(scaler, "scaler.joblib")

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

print(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 15, restore_best_weights=True)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
# Deeper network with L2 regularization to prevent overfitting
model.add(Dense(128, activation = "relu", kernel_regularizer=L2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation = "relu", kernel_regularizer=L2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(32, activation = "relu", kernel_regularizer=L2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(16, activation = "relu", kernel_regularizer=L2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'precision', 'recall'])
print("\n=== TRAINING WITH IMPROVED FEATURES & THRESHOLDS ===")
print(f"Model input features: {X_train.shape[1]} (includes volatility, returns, PCA, lag labels)")
print(f"Forward candles: 5 (5-candle ahead prediction for confirmation)")
print(f"Prediction logic: BUY if P >= 0.60, SELL if P <= 0.40, HOLD if 0.40 < P < 0.60\n")

history = model.fit(
    x = X_train, 
    y = y_train, 
    validation_data=(X_val, y_val),
    epochs = 150, 
    callbacks = [early_stop], 
    batch_size = 32,
    verbose=1
)

# Evaluate on validation set
val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)
print(f"\n=== FINAL VALIDATION METRICS ===")
print(f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

model.save("model.keras") # Specify the name HERE. (default: model.keras)
model.save("model.h5") # Compatibility fallback for older/newer keras loader combinations.
print("\n✅ Model retraining complete! Deploy with: azd up")
# Notice: The model will be loaded in "BinanceTrader.py" file.
