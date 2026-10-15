# TCS Stock Price Prediction Using Machine Learning

## 1. Project Overview

TCS Stock Price Prediction is a machine learning project developed to analyze historical stock market data of Tata Consultancy Services (TCS). The project uses Python and Flask to provide an interactive dashboard for visualizing stock prices and evaluating prediction models.

## 2. Objectives

- Analyze historical TCS stock price data.
- Visualize stock price trends over time.
- Predict the next day's closing price using Linear Regression.
- Compare Linear Regression with a baseline prediction method.
- Evaluate model performance using MAE and RMSE.

## 3. Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Frontend:** HTML and CSS
- **Deployment:** Render
- **Version Control:** Git and GitHub

## 4. Project Features

- Historical TCS stock price dashboard
- Stock closing price visualization
- Actual vs. Predicted stock price graph
- Linear Regression model
- Baseline model comparison
- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) evaluation

## 5. Project Structure

```text
TCS-Stock-Prediction/
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── data/
    └── tcs_stock_data.csv
```

## 6. Installation and Execution

### Prerequisites
- Python installed on your system
- Git (optional, for cloning the repository)

### Steps

1. Clone the repository:

   `git clone https://github.com/safiii05/TCS-Stock-Price-Prediction.git`

2. Navigate to the project folder:

   `cd TCS-Stock-Price-Prediction`

3. Install the required packages:

   `pip install -r requirements.txt`

4. Run the Flask application:

   `python app.py`

5. Open your browser and visit:

   `http://127.0.0.1:5000`

## 7. Model Evaluation

The project compares Linear Regression with a baseline model using two evaluation metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Measures prediction errors while penalizing larger errors more strongly.

In the current test results, the baseline model achieved lower MAE and RMSE than Linear Regression. This demonstrates the importance of evaluating machine learning models against a simple benchmark.

## 8. Results

The dashboard displays historical TCS closing prices, actual-versus-predicted prices, and a comparison of model performance
