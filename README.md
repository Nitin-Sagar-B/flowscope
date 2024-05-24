# FlowScope

### Enhancing Decision Making and Prediction Optimization using the HybridFlow Forecast Model

## Overview

FlowScope is an advanced web application designed to improve decision-making and prediction accuracy for web traffic analysis using a HybridFlow Forecast Model. The model combines several state-of-the-art time series forecasting techniques, including ARIMA, SARIMA, ETS, and LSTM, to deliver robust and accurate predictions. This project is particularly focused on the analysis and forecasting of web traffic, allowing businesses to make data-driven decisions to optimize their operations.

## Features

- **HybridFlow Forecast Model**: Integrates multiple forecasting models (ARIMA, SARIMA, ETS, LSTM) to enhance prediction accuracy.
- **Customizable Inputs**: Allows users to upload their dataset, specify relevant columns, and configure model parameters.
- **Interactive Dashboard**: Provides an intuitive and user-friendly interface for visualizing raw data, model predictions, and evaluation metrics.
- **Model Evaluation**: Offers comprehensive performance metrics (MAE, MSE, RMSE, MAPE) for both testing and future predictions.
- **Future Predictions**: Generates and visualizes future time series predictions based on the trained models.
- **Export Functionality**: Enables users to export the prediction results to a CSV file.

## Installation

To run FlowScope locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flowscope.git
   ```
2. Navigate to the project directory:
   ```bash
   cd flowscope
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the application in your web browser (usually http://localhost:8501).

## Application Flow

### Preprocessing Data

- **Data Upload**: Users upload their time-stamped dataset in CSV format.
- **Data Cleaning**: Missing values are filled with the mean of the column, and the data is sorted by the timestamp.

### Model Training and Predictions

1. **ARIMA**: Suitable for short-term forecasting of stationary data.
2. **SARIMA**: Ideal for capturing seasonal patterns and long-term trends.
3. **ETS**: Models error, trend, and seasonality without requiring differencing.
4. **LSTM**: Captures long-term dependencies and non-linear relationships in the data.

### Evaluation and Visualization

- **Model Evaluation**: Calculates MAE, MSE, RMSE, and MAPE for each model on the testing data.
- **Future Predictions**: Generates predictions for future time steps and evaluates model performance.
- **Visualizations**: Displays actual vs. predicted values and future predictions using interactive charts.

## Contributors

### Developers and Innovators

- **B Susheel**
  - [Email](mailto:21211a7205@bvrit.ac.in)
- **Nitin Sagar B**
  - [Email](mailto:21211a7207@bvrit.ac.in)
  - [GitHub](https://www.github.com/nitin-sagar-b/)
  - [LinkedIn](https://www.linkedin.com/in/nitin-sagar-boyeena/)
- **Md Reshma**
  - [Email](mailto:21211a7243@bvrit.ac.in)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was developed as part of a research initiative at BVRIT Hyderabad College of Engineering for Women. Special thanks to our mentors and colleagues for their support and guidance.

---
