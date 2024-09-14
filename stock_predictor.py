import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Load stock data
df = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[['Close']].values)

# Prepare data for LSTM
def create_dataset(dataset, look_back=60):
    x, y = [], []
    for i in range(len(dataset) - look_back):
        x.append(dataset[i:i+look_back])
        y.append(dataset[i+look_back])
    return np.array(x), np.array(y)

look_back = 60
x_train, y_train = create_dataset(df_scaled)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64)
model.save('stock_lstm_model.h5')
