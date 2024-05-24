import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

#function to preprocess the dataset using pandas
def preprocess_data(dataset):
    dataset.fillna(dataset.mean(), inplace=True)
    dataset = dataset.sort_index()

    return dataset

#function to split the dataset into training and testing sets
def train_test_split(dataset, split_ratio=0.8):
    split_point = int(len(dataset) * split_ratio)
    train_data = dataset.iloc[:split_point]
    test_data = dataset.iloc[split_point:]
    return train_data, test_data

#fitting ARIMA model and make predictions
def fit_arima_model(train_data, order):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

#fitting SARIMA model and make predictions
def fit_sarima_model(train_data, order, seasonal_order):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

#fitting ETS model and make predictions
def fit_ets_model(train_data, seasonal, seasonal_periods):
    model = ETSModel(train_data, error="add", trend="add", seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    return model_fit

#fitting LSTM model
def fit_lstm_model(train_data, n_steps):
    X, y = [], []
    for i in range(len(train_data) - n_steps):
        X.append(train_data[i:i + n_steps])
        y.append(train_data[i + n_steps])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    return model

#generating predictions using a model
def generate_predictions(model_fit, test_data, steps):
    predictions = model_fit.forecast(steps=steps)
    return predictions

#generating LSTM future predictions
def generate_lstm_future_predictions(model, last_window, future_steps):
    future_predictions = []
    for i in range(future_steps):
        prediction = model.predict(last_window.reshape(1, -1, 1))
        future_predictions.append(prediction[0][0])
        last_window = np.append(last_window[1:], prediction[0][0])
    return future_predictions

#model performance
def evaluate_model(actual, predictions):
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    evaluation_metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
    }
    return evaluation_metrics

#main function of the dashboard
def main():

    st.set_page_config(
        page_title="FlowScope",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Custom CSS style
    header_style = """
    <style>
    .stApp {
        background-color: black; 
        color: white; 
        font-family: 
    }
    </style>
    """
    st.markdown(header_style, unsafe_allow_html=True)

    # Set the title and subheader for web traffic analysis
    st.title('FlowScope')
    st.subheader('Enhancing Decision Making and Prediction Optimization using HybridFlow Forecast Model')
    with st.sidebar:
        # Add elements to the sidebar
        st.subheader('HybridFlow Forecast')
        st.write("This Hybrid model includes the following Individual models:")
        with st.expander("ARIMA - Auto Regressive Integrated Moving Average"):
            st.write("- Excellent for short-term forecasting of stationary time series data with clear and consistent patterns.")
            st.write("- Performs well when there is a strong linear relationship between past and present observations.")
        with st.expander("SARIMA - Seasonal Auto Regressive Integrated Moving Average"):
            st.write("- Highly effective for forecasting seasonal time series data with recurring patterns and long-term trends.")
            st.write("- Performs well when there are seasonal effects in the data that need to be captured.")
        with st.expander("ETS - Errors Trends and Seasonality"):
            st.write("- Ideal for capturing both trend and seasonal components in time series data without requiring differencing.")
            st.write("- Performs well when dealing with data that exhibits changing error variances over time.")
        with st.expander("LSTM - Long Short Term Memory"):
            st.write("- Excellent for time series forecasting with long-term dependencies and complex patterns, such as natural language processing tasks.")
            st.write("- Performs well when there is a need to capture non-linear relationships in the data and handle long-range dependencies.")
        
        # Custom CSS style for center-aligned button, email links, and column data
        center_content = """
        <style>
        div.stButton > button, div.email-link, div.column-content {
            display: block;
            margin: 0 auto;
            text-align: center;
        }
        </style>
        """

        if st.button("Developers and Innovators"):
            col1, col2, col3 = st.columns(3)

            email1 = "21211a7205@bvrit.ac.in"
            email2 = "21211a7207@bvrit.ac.in"
            email3 = "21211a7243@bvrit.ac.in"
            git1 = ""
            git2 = "https://www.github.com/nitin-sagar-b/"
            git3 = ""
            lin1 = ""
            lin2 = "https://www.linkedin.com/in/nitin-sagar-boyeena/"
            lin3 = ""

            col1.markdown(f'<div class="column-content">B Susheel<br><a class="email-link" href="mailto:{email1}">Email</a><br><a class="email-link" href="{git1}">GitHub</a><br><a class="email-link" href="{lin1}">LinkedIN</a></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="column-content">Nitin Sagar B<br><a class="email-link" href="mailto:{email2}">Email</a><br><a class="email-link" href="{git2}">GitHub</a><br><a class="email-link" href="{lin2}">LinkedIN</a></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="column-content">Md Reshma<br><a class="email-link" href="mailto:{email3}">Email</a><br><a class="email-link" href="{git3}">GitHub</a><br><a class="email-link" href="{lin3}">LinkedIN</a></div>', unsafe_allow_html=True)

        st.markdown(center_content, unsafe_allow_html=True)

    #upload the dataset for web traffic analysis
    st.subheader("Upload your Time-Stamp Dataset below:")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Preprocess the dataset
        dataset = pd.read_csv(uploaded_file)

        #raw data display
        st.subheader("Raw Data")
        st.dataframe(dataset, height=300)

        #getting user input for datetime column and web traffic column
        datetime_column = st.text_input("Enter the column name for the datetime (timestamp) data:")
        web_traffic_column = st.text_input("Enter the column name for the web traffic data:")

        # Check if the user has entered valid column names
        if datetime_column == "" or web_traffic_column == "":
            st.info("Please enter the above column names to proceed further.")
            return

        # Check if the entered columns exist in the dataset
        if datetime_column not in dataset.columns or web_traffic_column not in dataset.columns:
            st.error("One or both of the entered column names do not exist in the dataset.")
            return

        # Convert the datetime column to datetime type and set it as the index
        dataset[datetime_column] = pd.to_datetime(dataset[datetime_column])
        dataset.set_index(datetime_column, inplace=True)

        # Use the user-provided web traffic column as the target column
        target_column = web_traffic_column

        # Split the dataset into training and testing sets
        train_data, test_data = train_test_split(dataset)

        # Model Order Selection for ARIMA (You can customize this based on your dataset)
        arima_order = (1, 1, 1)

        # Model Order Selection for SARIMA (You can customize this based on your dataset)
        sarima_order = (1, 1, 1)
        sarima_seasonal_order = (1, 1, 1, 12)  # Assuming the seasonal period is 12 (monthly data)

        # Model Selection for ETS (You can customize this based on your dataset)
        ets_seasonal = 'add'
        ets_seasonal_periods = 12  # Assuming the seasonal period is 12 (monthly data)

        # LSTM Configuration (You can customize this based on your dataset)
        lstm_n_steps = 10

        # Fit ARIMA model
        progress_message = st.empty()

        with st.spinner("Training ARIMA model..."):
            arima_model_fit = fit_arima_model(train_data[target_column], arima_order)
            time.sleep(0.5)
            progress_message.success("ARIMA model trained successfully!")

        with st.spinner("Training SARIMA model..."):
            sarima_model_fit = fit_sarima_model(train_data[target_column], sarima_order, sarima_seasonal_order)
            time.sleep(0.5)
            progress_message.success("SARIMA model trained successfully!")

        with st.spinner("Training ETS model..."):
            ets_model_fit = fit_ets_model(train_data[target_column], ets_seasonal, ets_seasonal_periods)
            time.sleep(0.5)
            progress_message.success("ETS model trained successfully!")

        with st.spinner("Training LSTM model..."):
            lstm_model = fit_lstm_model(train_data[target_column].values.reshape(-1, 1), lstm_n_steps)
            time.sleep(0.5)
            progress_message.success("LSTM models trained successfully!")
            time.sleep(0.5)
            progress_message.success("ALL models trained successfully!")

        # Generate predictions for each model
        with st.spinner("Generating predictions..."):
            arima_predictions = generate_predictions(arima_model_fit, test_data[target_column], len(test_data))
            sarima_predictions = generate_predictions(sarima_model_fit, test_data[target_column], len(test_data))
            ets_predictions = generate_predictions(ets_model_fit, test_data[target_column], len(test_data))

            lstm_predictions = []
            for i in range(len(test_data) - lstm_n_steps + 1):
                lstm_input = test_data[target_column].values[i:i + lstm_n_steps].reshape(1, -1, 1)
                lstm_prediction = lstm_model.predict(lstm_input)
                lstm_predictions.append(lstm_prediction[0][0])

            lstm_predictions = np.array(lstm_predictions)
        progress_message.success("Predictions generated successfully!")

        # Evaluate model performance for testing data
        arima_evaluation = evaluate_model(test_data[target_column], arima_predictions)
        sarima_evaluation = evaluate_model(test_data[target_column], sarima_predictions)
        ets_evaluation = evaluate_model(test_data[target_column], ets_predictions)
        lstm_evaluation = evaluate_model(test_data[target_column][lstm_n_steps - 1:], lstm_predictions)

        # Display the testing dataset and predictions for all models
        st.subheader("Testing Data vs. Model Predictions")

        # Combine the actual and predicted values into a single DataFrame
        combined_df = pd.DataFrame({
            'Actual': test_data[target_column],
            'ARIMA Predicted': arima_predictions,
            'SARIMA Predicted': sarima_predictions,
            'ETS Predicted': ets_predictions,
            'LSTM Predicted': np.concatenate((np.full(lstm_n_steps - 1, np.nan), lstm_predictions)),
        })

        # Visualization section
        st.line_chart(combined_df)

        # Display evaluation metrics for all models before showing future predictions
        st.subheader("Model Evaluation Metrics for Testing Data")

        col1, col2 = st.columns(2)  # Split the screen into two columns

        # ARIMA Evaluation Metrics
        with col1:
            st.write("ARIMA Evaluation Metrics:")
            st.write(arima_evaluation)

        # LSTM Evaluation Metrics
        with col2:
            st.write("LSTM Evaluation Metrics:")
            st.write(lstm_evaluation)

        # SARIMA Evaluation Metrics
        with col1:
            st.write("SARIMA Evaluation Metrics:")
            st.write(sarima_evaluation)

        # ETS Evaluation Metrics
        with col2:
            st.write("ETS Evaluation Metrics:")
            st.write(ets_evaluation)
        
        testing_models = {
        'ARIMA': arima_evaluation['RMSE'],
        'SARIMA': sarima_evaluation['RMSE'],
        'ETS': ets_evaluation['RMSE'],
        'LSTM': lstm_evaluation['RMSE'],
        }
        best_testing_model = min(testing_models, key=testing_models.get)
        st.write(f"The best model for testing data is: {best_testing_model}")

        # Make predictions for time steps beyond the last time step of the testing dataset
        st.subheader("Future Predictions")

        future_time_steps = st.number_input("Enter the number of future time steps to predict:", value=30, min_value=1, max_value=300)

        # Generate future predictions for each model
        future_arima_predictions = generate_predictions(arima_model_fit, dataset[target_column], future_time_steps)
        future_sarima_predictions = generate_predictions(sarima_model_fit, dataset[target_column], future_time_steps)
        future_ets_predictions = generate_predictions(ets_model_fit, dataset[target_column], future_time_steps)

        # Generate future predictions for LSTM
        last_window = dataset[target_column].values[-lstm_n_steps:]
        future_lstm_predictions = generate_lstm_future_predictions(lstm_model, last_window, future_time_steps)

        # Combine future predictions into a single DataFrame
        future_df = pd.DataFrame({
            'ARIMA Predicted': future_arima_predictions,
            'SARIMA Predicted': future_sarima_predictions,
            'ETS Predicted': future_ets_predictions,
            'LSTM Predicted': future_lstm_predictions,
        })

        # Visualization of future predictions
        st.line_chart(future_df)
        st.dataframe(future_df)

        # Evaluate model performance for future predictions
        future_arima_evaluation = evaluate_model(dataset[target_column][-future_time_steps:], future_arima_predictions)
        future_sarima_evaluation = evaluate_model(dataset[target_column][-future_time_steps:], future_sarima_predictions)
        future_ets_evaluation = evaluate_model(dataset[target_column][-future_time_steps:], future_ets_predictions)
        future_lstm_evaluation = evaluate_model(dataset[target_column][-future_time_steps:], future_lstm_predictions)

        # Display evaluation metrics for future predictions
        st.subheader("Model Evaluation Metrics for Future Predictions")

        col1, col2 = st.columns(2)  # Split the screen into two columns

        # ARIMA Evaluation Metrics for Future Predictions
        with col1:
            st.write("ARIMA Evaluation Metrics for Future Predictions:")
            st.write(future_arima_evaluation)

        # SARIMA Evaluation Metrics for Future Predictions
        with col1:
            st.write("SARIMA Evaluation Metrics for Future Predictions:")
            st.write(future_sarima_evaluation)

        # LSTM Evaluation Metrics for Future Predictions
        with col2:
            st.write("LSTM Evaluation Metrics for Future Predictions:")
            st.write(future_lstm_evaluation)

        # ETS Evaluation Metrics for Future Predictions
        with col2:
            st.write("ETS Evaluation Metrics for Future Predictions:")
            st.write(future_ets_evaluation)

        future_models = {
        'ARIMA': future_arima_evaluation['RMSE'],
        'SARIMA': future_sarima_evaluation['RMSE'],
        'ETS': future_ets_evaluation['RMSE'],
        'LSTM': future_lstm_evaluation['RMSE'],
        }
        best_future_model = min(future_models, key=future_models.get)
        st.write(f"The best model for future predictions is: {best_future_model}")

        # Exporting predictions to CSV
        if st.button("Export Predictions to CSV"):
            with st.spinner("Exporting predictions to CSV..."):
                combined_df.to_csv("predictions.csv", index=False)
                time.sleep(1)
            st.success("Predictions exported to predictions.csv!")

    # Custom CSS style for the header
    header_style = """
    <style>
    .footer {
    
        padding: 10px;
        color: grey;
        text-align: center;
        font-size: 20px;
        font-family: 'Trebuchet MS', sans-serif;
        font-style: dim;
    }
    </style>
    """

    # Display the header
    st.markdown(header_style, unsafe_allow_html=True)
    st.markdown("<div class='footer'>Unleash the power of Enhanced Prediction Using the HybridFlow Model</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
