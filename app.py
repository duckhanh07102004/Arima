from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Tải dữ liệu từ Yahoo Finance theo mã chứng khoán và ngày bắt đầu, kết thúc
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            return render_template('index.html', error="Không tìm thấy dữ liệu cho mã chứng khoán này.")

        # Xây dựng mô hình ARIMA
        model = ARIMA(data['Close'], order=(5, 1, 0))  # Tham số (p, d, q)
        model_fit = model.fit()

        # Dự đoán trong khoảng thời gian dữ liệu
        y_predicted = model_fit.predict(start=data.index[0], end=data.index[-1])

        # Tính RMSE
        rmse = np.sqrt(mean_squared_error(data['Close'], y_predicted))

        # Tạo biểu đồ cho giá thực tế và giá dự đoán
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Giá thực tế', color='blue')
        plt.plot(data.index, y_predicted, label='Giá dự đoán ARIMA', color='orange')
        plt.title(f'Dự đoán giá chứng khoán cho {ticker} với ARIMA')
        plt.xlabel('Ngày')
        plt.ylabel('Giá')
        plt.legend()
        plt.grid()
        plt.savefig('static/predicted_price_plot.png')  # Lưu biểu đồ dự đoán
        plt.close()

        return render_template('result.html', ticker=ticker, rmse=rmse,
                               predicted_plot_url='static/predicted_price_plot.png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
