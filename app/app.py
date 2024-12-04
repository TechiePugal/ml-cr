from flask import Flask, render_template, request
from flask_socketio import SocketIO
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import time

app = Flask(__name__)
socketio = SocketIO(app)  # Initialize SocketIO

crypto_currencies = ['BTC', 'ETH', 'LTC']
against_currency = 'INR'


def create_and_fit_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Simulate training process with epoch updates to the client
    for epoch in range(1, 26):  # Simulating 25 epochs
        time.sleep(0.5)  # Simulate training time
        loss = np.random.random()  # Replace this with actual loss from model
        socketio.emit('training_update', {'epoch': epoch, 'loss': f"{loss:.4f}"})
    return model


def forecast_prices(crypto_currency, target_date):
    start_date = dt.datetime(2018, 1, 1)
    end_date = dt.datetime.now()
    df = yf.download(f'{crypto_currency}-{against_currency}', start=start_date, end=end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']].values)

    # Prepare training data
    lookback = 60
    x_train, y_train = [], []
    for i in range(lookback, len(df_scaled)):
        x_train.append(df_scaled[i - lookback:i, 0])
        y_train.append(df_scaled[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create and fit model
    model = create_and_fit_model(x_train, y_train)

    # Predict next 7 days
    model_inputs = df_scaled[len(df_scaled) - lookback:].reshape(-1, 1)
    predictions = []
    for _ in range(7):
        model_input = model_inputs[-lookback:].reshape((1, lookback, 1))
        prediction = model.predict(model_input)
        predictions.append(prediction[0, 0])
        model_inputs = np.append(model_inputs, prediction)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = [target_date + dt.timedelta(days=i) for i in range(7)]

    results = pd.DataFrame({
        'Cryptocurrency': crypto_currency,
        'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
        'Predicted Price (INR)': [round(price[0], 2) for price in predicted_prices]
    })

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        target_date = request.form.get("target_date")
        target_date = dt.datetime.strptime(target_date, "%Y-%m-%d")

        all_results = []
        for crypto_currency in crypto_currencies:
            result = forecast_prices(crypto_currency, target_date)
            all_results.append(result)

        final_df = pd.concat(all_results, ignore_index=True)
        return final_df.to_html()  # Return HTML table for demonstration

    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, debug=True)
