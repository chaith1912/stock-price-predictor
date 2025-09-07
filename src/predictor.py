import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===============================
# STEP 1: Download Stock Data
# ===============================
print("Downloading stock data...")
data = yf.download("AAPL", start="2015-01-01", end="2023-12-31")
data.to_csv("../data/apple_stock.csv")
print("Data saved to data/apple_stock.csv")

# ===============================
# STEP 2: Linear Regression Model
# ===============================
print("\nRunning Linear Regression model...")
df = data[['Close']].reset_index()
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

X = np.array(df[['Close']])
y = np.array(df['Target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(y_test, lr_predictions))

plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Price")
plt.plot(lr_predictions, label="LR Predicted Price")
plt.title("Linear Regression Prediction")
plt.legend()
plt.show()

# ===============================
# STEP 3: LSTM Model
# ===============================
print("\nRunning LSTM model...")
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(np.array(df['Close']).reshape(-1,1))

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_lstm, y_lstm = create_dataset(scaled, time_step)
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

train_size = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step,1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=64, verbose=1)

lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

plt.figure(figsize=(10,5))
plt.plot(scaler.inverse_transform(y_test_lstm.reshape(-1,1)), label="Actual Price")
plt.plot(lstm_predictions, label="LSTM Predicted Price")
plt.title("LSTM Prediction")
plt.legend()
plt.show()
