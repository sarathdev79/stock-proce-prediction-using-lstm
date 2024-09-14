import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def load_stock_data(ticker, start_date, end_date):
    """
    Loads historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Stock price data.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def scale_data(df, feature='Close'):
    """
    Scales the specified feature of the stock data between 0 and 1.
    
    Args:
        df (pd.DataFrame): The stock data DataFrame.
        feature (str): The column name to be scaled (default is 'Close').
    
    Returns:
        np.array: Scaled feature values.
        MinMaxScaler: Scaler object to inverse transform the predictions later.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[[feature]].values)
    return df_scaled, scaler

def create_dataset(dataset, look_back=60):
    """
    Converts the time series data into sequences for LSTM input.
    
    Args:
        dataset (np.array): The scaled stock data.
        look_back (int): The number of previous time steps to use as input features.
    
    Returns:
        np.array: Input features for the LSTM model.
        np.array: Target variable for the LSTM model.
    """
    x, y = [], []
    for i in range(len(dataset) - look_back):
        x.append(dataset[i:i + look_back])
        y.append(dataset[i + look_back])
    return np.array(x), np.array(y)

def load_and_prepare_data(ticker='AAPL', start_date='2020-01-01', end_date='2023-01-01', look_back=60):
    """
    Loads stock data, scales it, and prepares it for the LSTM model.
    
    Args:
        ticker (str): Stock ticker symbol (default is 'AAPL').
        start_date (str): Start date for data retrieval (default is '2020-01-01').
        end_date (str): End date for data retrieval (default is '2023-01-01').
        look_back (int): Number of previous time steps for input (default is 60).
    
    Returns:
        np.array: Prepared input features for the LSTM model.
        np.array: Target variable for the LSTM model.
        MinMaxScaler: Scaler object to inverse transform the predictions later.
    """
    df = load_stock_data(ticker, start_date, end_date)
    df_scaled, scaler = scale_data(df)
    x_train, y_train = create_dataset(df_scaled, look_back)
    
    return x_train, y_train, scaler

if __name__ == "__main__":
    x_train, y_train, scaler = load_and_prepare_data(ticker='AAPL', start_date='2020-01-01', end_date='2023-01-01', look_back=60)
    print(f"Training data shape: {x_train.shape}")
