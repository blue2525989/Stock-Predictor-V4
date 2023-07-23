import argparse
import util.install
import util.data
import util.model
import util.update


def compare_predictions():
    print("Comparing the predictions with the actual data...")
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Get a list of CSV files in the "data" folder
    data_folder = "data"
    csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]

    # Display the list of CSV files to the user
    print("Available CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")

    # Ask the user to select a CSV file
    selected_file = None
    while selected_file is None:
        try:
            file_number = int(
                input(
                    "Enter the number corresponding to the CSV file you want to select: "
                )
            )
            if file_number < 1 or file_number > len(csv_files):
                raise ValueError()
            selected_file = csv_files[file_number - 1]
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Load predicted and actual data
    predicted_data = pd.read_csv("predictions.csv")
    actual_data = pd.read_csv(os.path.join(data_folder, selected_file))

    # Rename columns for clarity
    predicted_data = predicted_data.rename(columns={"Predicted Close": "Close"})
    actual_data = actual_data.rename(columns={"Close": "Actual Close"})

    # Join predicted and actual data on the date column
    combined_data = pd.merge(predicted_data, actual_data, on="Date")

    # Calculate the absolute percentage error between the predicted and actual values
    combined_data["Absolute % Error"] = (
        abs(combined_data["Close"] - combined_data["Actual Close"])
        / combined_data["Actual Close"]
        * 100
    )

    # Calculate the mean absolute percentage error and print it
    mape = combined_data["Absolute % Error"].mean()
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Find the row with the highest and lowest absolute percentage error and print them
    min_error_row = combined_data.iloc[combined_data["Absolute % Error"].idxmin()]
    max_error_row = combined_data.iloc[combined_data["Absolute % Error"].idxmax()]
    print(f"\nMost Accurate Prediction:\n{min_error_row}\n")
    print(f"Least Accurate Prediction:\n{max_error_row}\n")

    # Plot the predicted and actual close prices
    plt.plot(combined_data["Date"], combined_data["Close"], label="Predicted Close")
    plt.plot(combined_data["Date"], combined_data["Actual Close"], label="Actual Close")

    # Add title and legend
    plt.title("Predicted vs Actual Close Prices")
    plt.legend()

    # Show plot
    plt.show()


def gen_stock():
    print("Generating Stock Data...")
    import csv
    import random
    import datetime

    def generate_stock_data_csv(file_path, num_lines, data_type):
        # Define the column names
        columns = ["Date", "Close", "Adj Close", "Volume", "High", "Low", "Open"]

        # Generate stock data based on the selected data type
        data = []
        start_date = datetime.datetime(2022, 1, 1)
        for i in range(num_lines):
            if data_type == "linear":
                close_price = 100.0 + i * 10
            elif data_type == "exponential":
                close_price = 100.0 * (1.1**i)
            elif data_type == "random":
                if i > 0:
                    prev_close = data[i - 1][1]
                    close_price = prev_close * random.uniform(0.95, 1.05)
                else:
                    close_price = 100.0
            elif data_type == "trend":
                if i > 0:
                    prev_close = data[i - 1][1]
                    close_price = prev_close + random.uniform(-2, 2)
                else:
                    close_price = 100.0
            else:
                raise ValueError("Invalid data type provided.")

            date = start_date + datetime.timedelta(days=i)
            adj_close = close_price
            volume = random.randint(100000, 1000000)
            high = close_price * random.uniform(1.01, 1.05)
            low = close_price * random.uniform(0.95, 0.99)
            open_price = close_price * random.uniform(0.98, 1.02)

            data.append(
                [
                    date.strftime("%Y-%m-%d"),
                    close_price,
                    adj_close,
                    volume,
                    high,
                    low,
                    open_price,
                ]
            )

        # Save the generated data to a CSV file
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(data)

        print(f"Stock data CSV file '{file_path}' generated successfully.")

    # Prompt the user for options
    num_lines = int(input("Enter the number of lines: "))
    data_type = input("Enter the data type (linear/exponential/random/trend): ")
    file_path = "data/example.csv"

    # Generate the stock data CSV file
    generate_stock_data_csv(file_path, num_lines, data_type)


def do_all_actions():
    util.data.prepare_data()
    util.model.train_model()
    util.model.evaluate_model()
    util.model.fine_tune_model()
    util.model.evaluate_model()
    util.data.predict_future_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPV4 Script")
    parser.add_argument(
        "--update", action="store_true", help="Check updates for SPV4"
    )
    parser.add_argument(
        "--install", action="store_true", help="Install all dependencies for SPV4"
    )
    parser.add_argument(
        "--generate_stock", action="store_true", help="Generate Stock Data"
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Preprocess and Prepare the CSV Data",
    )
    parser.add_argument("--train", action="store_true", help="Train the SPV4 Model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the Model")
    parser.add_argument("--fine_tune", action="store_true", help="Finetune the Model")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Utilize the Model for Predicting Future Data (30 Days)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the Predictions with the Actual Data",
    )
    parser.add_argument(
        "--do-all",
        action="store_true",
        help="Do all actions from above (No Install & Generating Stock Data)",
    )

    args = parser.parse_args()

    if args.do_all:
        do_all_actions()
    else:
        if args.install:
            util.install.install_dependencies()
        if args.update:
            util.update.update()
        if args.generate_stock:
            gen_stock()
        if args.prepare_data:
            util.data.prepare_data()
        if args.train:
            util.model.train_model()
        if args.eval:
            util.model.evaluate_model()
        if args.fine_tune:
            util.model.fine_tune_model()
        if args.predict:
            util.data.predict_future_data()
        if args.compare:
            compare_predictions()