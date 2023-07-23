def prepare_data():
    print("Preprocessing and preparing the CSV data...")
    import os
    import pandas as pd
    import talib
    import matplotlib.pyplot as plt

    # List all CSV files in the "data" folder
    data_folder = os.path.join("data", "unrefined")
    csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]

    # Print the available CSV files with numbers for selection
    print("Available CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")

    # Ask for user input to select a CSV file
    selected_file_index = (
        int(input("Enter the number of the CSV file to preprocess: ")) - 1
    )
    selected_file = csv_files[selected_file_index]
    file_prefix = selected_file.replace('.csv', '')
    print(selected_file)
    print(file_prefix)
    selected_file_path = os.path.join(data_folder, selected_file)

    # Preprocess the selected CSV file
    df = pd.read_csv(selected_file_path)

    df["SMA"] = talib.SMA(df["Close"], timeperiod=14)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["MACD"], _, _ = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["upper_band"], df["middle_band"], df["lower_band"] = talib.BBANDS(
        df["Close"], timeperiod=20
    )
    df["aroon_up"], df["aroon_down"] = talib.AROON(df["High"], df["Low"], timeperiod=25)
    df["kicking"] = talib.CDLKICKINGBYLENGTH(
        df["Open"], df["High"], df["Low"], df["Close"]
    )

    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["upper_band_supertrend"] = df["High"] - (df["ATR"] * 2)
    df["lower_band_supertrend"] = df["Low"] + (df["ATR"] * 2)
    df["in_uptrend"] = df["Close"] > df["lower_band_supertrend"]
    df["supertrend_signal"] = df["in_uptrend"].diff().fillna(0)

    # Replace "False" with 0 and "True" with 1
    df = df.replace({False: 0, True: 1})

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Concatenate the columns in the order you want
    df2 = pd.concat(
        [
            df["Date"],
            df["Close"],
            df["Adj Close"],
            df["Volume"],
            df["High"],
            df["Low"],
            df["SMA"],
            df["MACD"],
            df["upper_band"],
            df["middle_band"],
            df["lower_band"],
            df["supertrend_signal"],
            df["RSI"],
            df["aroon_up"],
            df["aroon_down"],
            df["kicking"],
            df["upper_band_supertrend"],
            df["lower_band_supertrend"],
        ],
        axis=1,
    )

    #TODO make unique name of data.csv, prefix with ticket name
    # Save the DataFrame to a new CSV file with indicators
    df2.to_csv(os.path.join(os.path.join("data", "refined"), file_prefix + "-data.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df["Close"], label="Close")
    ax1.plot(df["SMA"], label="SMA")
    ax1.fill_between(
        df.index, df["upper_band"], df["lower_band"], alpha=0.2, color="gray"
    )
    ax1.plot(df["upper_band"], linestyle="dashed", color="gray")
    ax1.plot(df["middle_band"], linestyle="dashed", color="gray")
    ax1.plot(df["lower_band"], linestyle="dashed", color="gray")
    ax1.scatter(
        df.index[df["supertrend_signal"] == 1],
        df["Close"][df["supertrend_signal"] == 1],
        marker="^",
        color="green",
        s=100,
    )
    ax1.scatter(
        df.index[df["supertrend_signal"] == -1],
        df["Close"][df["supertrend_signal"] == -1],
        marker="v",
        color="red",
        s=100,
    )
    ax1.legend()

    ax2.plot(df["RSI"], label="RSI")
    ax2.plot(df["aroon_up"], label="Aroon Up")
    ax2.plot(df["aroon_down"], label="Aroon Down")
    ax2.scatter(
        df.index[df["kicking"] == 100],
        df["High"][df["kicking"] == 100],
        marker="^",
        color="green",
        s=100,
    )
    ax2.scatter(
        df.index[df["kicking"] == -100],
        df["Low"][df["kicking"] == -100],
        marker="v",
        color="red",
        s=100,
    )
    ax2.legend()

    plt.xlim(df.index[0], df.index[-1])

    plt.show()

def predict_future_data():
    print("Utilizing the model for predicting future data (30 days)...")
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt


    # List all CSV files in the "data" folder
    data_folder = os.path.join("data", "refined")
    csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]

    print("Available CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")

    # Ask for user input to select a CSV file
    selected_file_index = (
        int(input("Enter the number of the CSV file to process: ")) - 1
    )
    selected_file = csv_files[selected_file_index]

    # Load data
    data = pd.read_csv(selected_file)

    # Normalize data
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(
        data[
            [
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",
            ]
        ]
    )

    # Define time steps
    timesteps = 100

    # Create sequences of timesteps
    def create_sequences(data, timesteps):
        X = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
        return np.array(X)

    X_data = create_sequences(data_norm, timesteps)

    # Load model
    model = load_model("model.h5")
    model.summary()

    num_predictions = 30

    # Make predictions for next num_predictions days
    X_pred = X_data[-num_predictions:].reshape(
        (num_predictions, timesteps, X_data.shape[2])
    )
    y_pred = model.predict(X_pred)[:, 0]

    # Inverse transform predictions
    y_pred = scaler.inverse_transform(
        np.hstack(
            [
                np.zeros((len(y_pred), data_norm.shape[1] - 1)),
                np.array(y_pred).reshape(-1, 1),
            ]
        )
    )[:, -1]

    # Generate date index for predictions
    last_date = data["Date"].iloc[-1]
    index = pd.date_range(
        last_date, periods=num_predictions, freq="D", tz="UTC"
    ).tz_localize(None)

    # Calculate % change
    y_pred_pct_change = (y_pred - y_pred[0]) / y_pred[0] * 100

    # Save predictions and % change in a CSV file
    predictions = pd.DataFrame(
        {"Date": index, "Predicted Close": y_pred, "% Change": y_pred_pct_change}
    )
    predictions.to_csv("predictions.csv", index=False)

    print(predictions)

    # Find the rows with the lowest and highest predicted close and the highest and lowest % change
    min_close_row = predictions.iloc[predictions["Predicted Close"].idxmin()]
    max_close_row = predictions.iloc[predictions["Predicted Close"].idxmax()]
    max_pct_change_row = predictions.iloc[predictions["% Change"].idxmax()]
    min_pct_change_row = predictions.iloc[predictions["% Change"].idxmin()]

    # Print the rows with the lowest and highest predicted close and the highest and lowest % change
    print(f"\n\nHighest predicted close:\n{max_close_row}\n")
    print(f"Lowest predicted close:\n{min_close_row}\n")
    print(f"Highest % change:\n{max_pct_change_row}\n")
    print(f"Lowest % change:\n{min_pct_change_row}")

    # Plot historical data and predictions
    plt.plot(data["Close"].values, label="Actual Data")
    plt.plot(
        np.arange(len(data), len(data) + num_predictions),
        y_pred,
        label="Predicted Data",
    )

    # Add red and green arrows for highest and lowest predicted close respectively, and highest and lowest percentage change
    plt.annotate(
        "↓",
        xy=(min_close_row.name - len(data), min_close_row["Predicted Close"]),
        color="red",
        fontsize=16,
        arrowprops=dict(facecolor="red", shrink=0.05),
    )
    plt.annotate(
        "↑",
        xy=(max_close_row.name - len(data), max_close_row["Predicted Close"]),
        color="green",
        fontsize=16,
        arrowprops=dict(facecolor="green", shrink=0.05),
    )
    plt.annotate(
        "↑",
        xy=(max_pct_change_row.name - len(data), y_pred.max()),
        color="green",
        fontsize=16,
        arrowprops=dict(facecolor="green", shrink=0.05),
    )
    plt.annotate(
        "↓",
        xy=(min_pct_change_row.name - len(data), y_pred.min()),
        color="red",
        fontsize=16,
        arrowprops=dict(facecolor="red", shrink=0.05),
    )

    # Add legend and title
    plt.legend()
    plt.title("Predicted Close Prices")

    # Show plot
    plt.show()