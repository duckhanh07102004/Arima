import streamlit as st
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Streamlit App
st.title("Stock Price Prediction with ARIMA")

# Input from user
ticker = st.text_input("Enter stock ticker:", "AAPL")
start_date = st.date_input("Start date")
end_date = st.date_input("End date")

if st.button('Predict'):
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for this ticker.")
    else:
        # ARIMA Model
        model = ARIMA(data['Close'], order=(5, 1, 0))  # (p, d, q)
        model_fit = model.fit()

        # Predictions
        y_predicted = model_fit.predict(start=data.index[0], end=data.index[-1])

        # RMSE
        rmse = np.sqrt(mean_squared_error(data['Close'], y_predicted))

        # Plot actual vs predicted
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Actual Price', color='blue')
        ax.plot(data.index, y_predicted, label='Predicted Price (ARIMA)', color='orange')
        ax.set_title(f'Stock Price Prediction for {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid()

        # Display plot
        st.pyplot(fig)

        # Display RMSE
        st.write(f"Root Mean Square Error (RMSE): {rmse}")
