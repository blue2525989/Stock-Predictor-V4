def train_model():
    import os
    import sys

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        BatchNormalization,
        Conv1D,
        MaxPooling1D,
        TimeDistributed,
    )
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.metrics import mean_absolute_percentage_error, r2_score
    from optuna import create_study, Trial, visualization
    from optuna.samplers import TPESampler

    print("Training the SPV4 model...")
    print("TensorFlow version:", tf.__version__)

    # Define reward function
    def get_reward(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        reward = ((1 - mape) + r2) / 2
        return reward

    #TODO make unique name of data.csv, prefix with ticket name
    # Load data
    data = pd.read_csv("data.csv")

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

    # Split data into train and test sets
    train_data_norm = data_norm[: int(0.8 * len(data))]
    test_data_norm = data_norm[int(0.8 * len(data)):]

    # Define time steps
    timesteps = 100

    # Create sequences of timesteps
    def create_sequences(data, timesteps):
        X = []
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data_norm, timesteps)
    X_test, y_test = create_sequences(test_data_norm, timesteps)

    # Define the Deep RL model
    def create_model(trial):
        model = Sequential()
        model.add(
            Conv1D(
                filters=trial.suggest_int("filters", 50, 450),
                kernel_size=trial.suggest_int("kernel_size", 2, 15),
                activation="relu"
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(
            Conv1D(
                filters=trial.suggest_int("filters_2", 50, 450),
                kernel_size=trial.suggest_int("kernel_size_2", 1, 15),
                activation="relu"
            )
        )
        model.add(
            LSTM(
                units=trial.suggest_int("units", 50, 300),
                return_sequences=True,
                input_shape=(timesteps, X_train.shape[2])
            )
        )
        model.add(BatchNormalization())
        model.add(
            LSTM(
                units=trial.suggest_int("units_2", 50, 300),
                return_sequences=True
            )
        )
        model.add(BatchNormalization())
        model.add(Dense(units=trial.suggest_int("units_3", 50, 300)))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Dense(units=trial.suggest_int("units_4", 50, 300))))
        model.add(BatchNormalization())
        model.add(
            LSTM(
                units=trial.suggest_int("units_5", 25, 150),
                return_sequences=True
            )
        )
        model.add(BatchNormalization())
        model.add(
            LSTM(
                units=trial.suggest_int("units_6", 25, 150),
                return_sequences=True
            )
        )
        model.add(BatchNormalization())
        model.add(TimeDistributed(Dense(units=trial.suggest_int("units_7", 25, 150))))
        model.add(BatchNormalization())
        model.add(LSTM(units=trial.suggest_int("units_8", 12, 75)))
        model.add(BatchNormalization())
        model.add(Dense(units=1))

        # Define the RL optimizer and compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("learning_rate", 0.001, 0.015))
        model.compile(optimizer=optimizer, loss="mse")

        return model

    # Define RL training loop
    epochs = 10
    epochs1 = 3
    batch_size = 50
    best_reward = None

    def optimize_model(trial: Trial):
        model = create_model(trial)

        for i in range(epochs1):
            print("Epoch", i+1, "/", epochs1)
            # Train the model for one epoch
            for a in range(0, len(X_train), batch_size):
                if a == 0:
                    print(
                        "Batch", a+1, "/", len(X_train),
                        "(", ((a/len(X_train))*100), "% Done)"
                    )
                else:
                    sys.stdout.write('\033[F\033[K')
                    print(
                        "Batch", a+1, "/", len(X_train),
                        "(", ((a/len(X_train))*100), "% Done)"
                    )
                batch_X = X_train[a:a + batch_size]
                batch_y = y_train[a:a + batch_size]
                history = model.fit(
                    batch_X, batch_y,
                    batch_size=batch_size, epochs=1, verbose=0
                )
            sys.stdout.write('\033[F\033[K')
            sys.stdout.write('\033[F\033[K')

        # Evaluate the model on the test set
        y_pred_test = model.predict(X_test)
        test_reward = get_reward(y_test, y_pred_test)
        sys.stdout.write('\033[F\033[K')

        return test_reward

    # Define the search space boundaries and create Optuna study
    sampler = TPESampler(seed=42)
    study = create_study(sampler=sampler, direction="maximize")
    study.optimize(optimize_model, n_trials=5)

    # Get best parameters and reward
    best_trial = study.best_trial
    best_params = best_trial.params
    best_reward = best_trial.value

    print("\nBest reward:", best_reward)
    print("Best parameters:", best_params)

    # Load the best model and evaluate it
    model = create_model(best_trial)
    for i in range(epochs):
        print("Epoch", i+1, "/", epochs)
        # Train the model for one epoch
        for a in range(0, len(X_train), batch_size):
            if a == 0:
                print(
                    "Batch", a+1, "/", len(X_train),
                    "(", ((a/len(X_train))*100), "% Done)"
                )
            else:
                sys.stdout.write('\033[F\033[K')
                print(
                    "Batch", a+1, "/", len(X_train),
                    "(", ((a/len(X_train))*100), "% Done)"
                )
            batch_X = X_train[a:a + batch_size]
            batch_y = y_train[a:a + batch_size]
            history = model.fit(
                batch_X, batch_y,
                batch_size=batch_size, epochs=1, verbose=0
            )

        # Evaluate the model on the test set
        y_pred_test = model.predict(X_test)
        sys.stdout.write('\033[F\033[K')
        test_reward = get_reward(y_test, y_pred_test)

        print("Test reward:", test_reward)

        if i == 0:
            best_reward1 = test_reward

        if test_reward >= best_reward1:
            print("Model saved!")
            #TODO make unique name of model.h5, prefix with ticket name
            model.save("model.h5")

    if i == epochs - 1:
        #TODO make unique name of model.h5, prefix with ticket name
        model = load_model("model.h5")
        y_pred_test = model.predict(X_test)
        test_reward = get_reward(y_test, y_pred_test)
        test_loss = model.evaluate(X_test, y_test)

        print("Final test reward:", test_reward)
        print("Final test loss:", test_loss)

def evaluate_model():
    print("Evaluating the model...")
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    import matplotlib.pyplot as plt

    print("TensorFlow version:", tf.__version__)

    # Load data
    data = pd.read_csv("data.csv")

    # Split data into train and test sets
    train_data = data.iloc[: int(0.8 * len(data))]
    test_data = data.iloc[int(0.8 * len(data)) :]

    # Normalize data
    scaler = MinMaxScaler()
    train_data_norm = scaler.fit_transform(
        train_data[
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
    test_data_norm = scaler.transform(
        test_data[
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

    def create_sequences(data, timesteps):
        X = []
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # Load model
    model = load_model("model.h5")

    # Evaluate model
    rmse_scores = []
    mape_scores = []
    rewards = []
    model = load_model("model.h5")
    X_test, y_test = create_sequences(test_data_norm, timesteps)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse_scores.append(rmse)
    mape_scores.append(mape)
    rewards.append(1 - mape)

    # Print results
    print(f"Mean RMSE: {np.mean(rmse_scores)}")
    print(f"Mean MAPE: {np.mean(mape_scores)}")
    print(f"Total Reward: {sum(rewards)}")

def fine_tune_model():
    print("Finetuning the model...")
    import os
    import signal

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import sys
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
    )
    import matplotlib.pyplot as plt
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    print("TensorFlow version:", tf.__version__)

    # Define reward function
    def get_reward(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        reward = ((1 - mape) + r2) / 2
        return reward

    # Load data
    data = pd.read_csv("data.csv")

    # Split data into train and test sets
    train_data = data.iloc[: int(0.8 * len(data))]
    test_data = data.iloc[int(0.8 * len(data)) :]

    # Normalize data
    scaler = MinMaxScaler()
    train_data_norm = scaler.fit_transform(
        train_data[
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
    test_data_norm = scaler.transform(
        test_data[
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
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data_norm, timesteps)
    X_test, y_test = create_sequences(test_data_norm, timesteps)

    # Define reward threshold
    reward_threshold = float(
        input("Enter the reward threshold (0 - 1, 0.9 recommended): ")
    )

    # Initialize rewards
    rewards = []
    mses = []
    mapes = []
    r2s = []
    count = 0

    # Function to handle SIGINT signal (CTRL + C)
    def handle_interrupt(signal, frame):
        print("\nInterrupt received.")

        # Ask the user for confirmation
        user_input = input(
            f"Are you sure that you want to End the Program? (yes/no): "
        )

        if user_input.lower() == "yes":
            exit(0)

        else:
            print("Continuing the Fine-tuning Process")

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_interrupt)

    while True:
        # Load model
        model = load_model("model.h5")
        print("\nEvaluating Model")
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Append rewards
        reward = get_reward(y_test, y_pred)
        rewards.append(reward)
        mses.append(mse)
        mapes.append(mape)
        r2s.append(r2)

        # Print current rewards
        print("Rewards:", rewards)
        print("MAPE:", mape)
        print("MSE:", mse)
        print("R2:", r2)
        count += 1
        print("Looped", count, "times.")

        # Check if reward threshold is reached
        if len(rewards) >= 1 and sum(rewards[-1:]) >= reward_threshold:
            print("Reward threshold reached!")
            model.save("model.h5")

            break
        else:
            print("Training Model with 5 Epochs")
            epochs = 5
            batch_size = 50
            for i in range(epochs):
                print("Epoch", i, "/", epochs)
                # Train the model for one epoch
                for a in range(0, len(X_train), batch_size):
                    if a == 0:
                        print("Batch", a, "/", len(X_train), "(", ((a/len(X_train))*100), "% Done)")
                    else:
                        sys.stdout.write('\033[F\033[K')
                        print("Batch", a, "/", len(X_train), "(", ((a/len(X_train))*100), "% Done)")
                    batch_X = X_train[a:a + batch_size]
                    batch_y = y_train[a:a + batch_size]
                    history = model.fit(batch_X, batch_y, batch_size=batch_size, epochs=1, verbose=0)

                # Evaluate the model on the test set
                y_pred_test = model.predict(X_test)
                sys.stdout.write('\033[F\033[K')
                test_reward = get_reward(y_test, y_pred_test)

                print("Test reward:", test_reward)

                if i == 0 and count == 1:
                    best_reward1 = test_reward

                if test_reward >= best_reward1:
                    print("Model saved!")
                    model_saved = 1
                    best_reward1 = test_reward
                    model.save("model.h5")
                
                if test_reward >= reward_threshold:
                    print("Model reached reward threshold", test_reward, ". Saving and stopping epochs!")
                    model_saved = 1
                    model.save("model.h5")
                    break

