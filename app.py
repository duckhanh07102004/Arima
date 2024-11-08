import numpy as np
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import time

# Load tickers from CSV
@st.cache_data  # Cache this function to avoid reloading on every interaction
def load_ticker_list():
    tickers_df = pd.read_csv("symbols_valid_meta.csv")  # Load your own CSV file of tickers
    tickers_df["Display"] = tickers_df["Symbol"] + " - " + tickers_df["Security Name"]  # Combine symbol and name for display
    return tickers_df

# Streamlit configuration
st.title("Stock Price Prediction using ARIMA")
st.sidebar.header("User Inputs")

# Load tickers
tickers_df = load_ticker_list()
all_ticker_options = tickers_df["Display"].tolist()  # Options for display in dropdown
all_ticker_symbols = tickers_df["Symbol"].tolist()  # Actual symbols for data retrieval

# User selects a ticker by display name
selected_display = st.sidebar.selectbox("Choose Stock Ticker", options=all_ticker_options)

# Retrieve the corresponding symbol from the selected display
selected_symbol = tickers_df[tickers_df["Display"] == selected_display]["Symbol"].values[0]

# Additional user inputs
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today())
future_days = st.sidebar.slider("Number of Future Days to Predict", 1, 60, 30)
st.sidebar.subheader("ARIMA Model Configuration")
p = st.sidebar.number_input("Enter p (autoregressive term):", min_value=0, max_value=10, value=5)
d = st.sidebar.number_input("Enter d (difference term):", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("Enter q (moving average term):", min_value=0, max_value=10, value=0)


# Function to download data
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values, data.index  # Returns price data and dates

# Function to preprocess data
def preprocess_data(data, dates):
    if len(data) == 0:
        st.error("No data downloaded. Please check your ticker symbol or date range.")
        return None, None, None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Save the scaler for inverse transformation
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return scaled_data, scaler, dates

# Function to train ARIMA model
def train_model(data):
    model = ARIMA(data, order=(p, d, q))  # Adjust order if necessary
    arima_model = model.fit()
    
    # Calculate RMSE on training data
    fitted_values = arima_model.fittedvalues
    mse = mean_squared_error(data, fitted_values)  # Skip first value due to differencing
    rmse = np.sqrt(mse)
    st.write(f'RMSE on training data: {rmse}')
    
    # Save the trained ARIMA model
    with open('stock_arima_model.pkl', 'wb') as f:
        pickle.dump(arima_model, f)
    
    return arima_model

# Function to make predictions
def predict(model, future_days):
    forecast = model.forecast(steps=future_days)
    return forecast

# Function to plot results
def plot_results(data, forecast, dates, ticker):
    plt.figure(figsize=(12, 8))
    plt.plot(dates, data, label='True Prices', color='blue')
    
    # Create future dates for the forecast
    future_dates = pd.date_range(start=dates[-1], periods=len(forecast)+1, freq='B')[1:]
    plt.plot(future_dates, forecast, label='Future Predictions', linestyle='--', color='green')
    
    plt.legend()
    plt.title(f"{ticker} Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.grid(True)
    st.pyplot(plt.gcf())  # Use plt.gcf() to show the current figure in Streamlit

# Function to display future predictions
def print_future_predictions(future_predictions, end_date):
    st.write("\nPredicted Prices for the next days:")
    for i, prediction in enumerate(future_predictions, start=1):
        next_date = pd.Timestamp(end_date) + pd.DateOffset(days=i)
        st.write(f"{next_date.date()}: {prediction}")

# Main function
def main():
    # Download and preprocess data
    data, dates = download_data(selected_symbol, start_date, end_date)
    data, scaler, dates = preprocess_data(data, dates)
    
    if data is None:
        return
    if st.sidebar.button('Run ARIMA Model'):
        # Start calculating time
        start_time = time.time()
    # Train ARIMA model and calculate RMSE
        model = train_model(data)
        
        # Predict future prices
        future_predictions = predict(model, future_days=future_days)
        
        # Inverse transform the predictions to original scale
        future_pred = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        
       
        # Print and plot future predictions
        print_future_predictions(future_pred, dates[-1])
        plot_results(scaler.inverse_transform(data.reshape(-1, 1)).flatten(), future_pred, dates, selected_symbol)
        end_time = time.time()  # Kết thúc đo thời gian
        execution_time = end_time - start_time
        st.write(f"Execution time : {execution_time} s")

if __name__ == "__main__":
    main()
