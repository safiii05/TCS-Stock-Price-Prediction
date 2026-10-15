
from flask import Flask, render_template_string
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from io import BytesIO
import base64

app = Flask(__name__)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tcs_stock_data.csv"

FEATURES = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

def load_and_prepare_data():
    """Load and prepare historical TCS stock data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Date", "Open", "High", "Low", "Close", "Volume"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Date"] + FEATURES
    )

    df = df.sort_values("Date").reset_index(drop=True)

    # Target: next available trading day's closing price
    df["Next_Close"] = df["Close"].shift(-1)

    df = df.dropna(
        subset=["Next_Close"]
    ).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "The dataset needs more historical records."
        )

    return df


# --------------------------------------------------
# 3. CREATE CHART IMAGE
# --------------------------------------------------

def figure_to_base64(fig):
    """Convert a Matplotlib figure into a webpage image."""

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=110,
        bbox_inches="tight"
    )

    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# 4. TRAIN MODELS AND PREPARE DASHBOARD
# --------------------------------------------------

def create_dashboard_data():

    df = load_and_prepare_data()

    X = df[FEATURES]
    y = df["Next_Close"]

    # Chronological 80/20 split
    split_index = int(len(df) * 0.8)

    if split_index < 2 or split_index >= len(df):
        raise ValueError(
            "Not enough data for training and testing."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict test-set prices
    predictions = model.predict(X_test)

    # Baseline: next close equals today's close
    baseline_predictions = X_test["Close"].to_numpy()

    # Evaluate Linear Regression
    linear_mae = mean_absolute_error(
        y_test, predictions
    )

    linear_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    # Evaluate baseline
    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    # Load the latest available original record
    raw_df = pd.read_csv(DATA_PATH)

    raw_df["Date"] = pd.to_datetime(
        raw_df["Date"], utc=True
    )

    for col in FEATURES:
        raw_df[col] = pd.to_numeric(
            raw_df[col], errors="coerce"
        )

    raw_df = raw_df.dropna(
        subset=["Date"] + FEATURES
    )

    raw_df = raw_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    latest = raw_df.iloc[-1]

    latest_date = latest["Date"].strftime("%d %b %Y")
    latest_close = float(latest["Close"])

    # Estimate next available trading day's close
    latest_features = pd.DataFrame(
        [[latest[col] for col in FEATURES]],
        columns=FEATURES
    )

    estimated_next_close = float(
        model.predict(latest_features)[0]
    )

    # --------------------------------------------------
    # 5. HISTORICAL STOCK PRICE CHART
    # --------------------------------------------------

    chart_df = raw_df.tail(100)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        chart_df["Date"],
        chart_df["Close"],
        color="#2563eb",
        linewidth=2.3,
        label="TCS Closing Price"
    )

    ax.set_title(
        "TCS Stock Closing Price - Last 100 Records",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    chart_image = figure_to_base64(fig)

    # --------------------------------------------------
    # 6. ACTUAL VS PREDICTED CHART
    # --------------------------------------------------

    test_dates = df["Date"].iloc[split_index:]

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        test_dates,
        y_test.to_numpy(),
        color="#16a34a",
        linewidth=2,
        label="Actual Closing Price"
    )

    ax2.plot(
        test_dates,
        predictions,
        color="#9333ea",
        linewidth=2,
        linestyle="--",
        label="Linear Regression Prediction"
    )

    ax2.set_title(
        "Actual vs Predicted TCS Closing Prices",
        fontsize=15,
        fontweight="bold"
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (₹)")
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.legend()
    fig2.autofmt_xdate()
    fig2.tight_layout()

    actual_predicted_chart_image = figure_to_base64(fig2)

    # --------------------------------------------------
    # 7. MODEL COMPARISON
    # --------------------------------------------------

    model_comparison = [
        {
            "name": "Linear Regression",
            "mae": float(linear_mae),
            "rmse": float(linear_rmse)
        },
        {
            "name": "Baseline",
            "mae": float(baseline_mae),
            "rmse": float(baseline_rmse)
        }
    ]

    # Identify the model with lower MAE
    if linear_mae < baseline_mae:
        better_model = "Linear Regression"
    elif baseline_mae < linear_mae:
        better_model = "Baseline"
    else:
        better_model = "Both models have equal MAE"

    return {
        "latest_date": latest_date,
        "latest_close": latest_close,
        "estimated_next_close": estimated_next_close,
        "mae": float(linear_mae),
        "rmse": float(linear_rmse),
        "total_records": len(raw_df),
        "chart_image": chart_image,
        "actual_predicted_chart_image":
            actual_predicted_chart_image,
        "model_comparison": model_comparison,
        "better_model": better_model
    }


# --------------------------------------------------
# 8. DASHBOARD HTML AND CSS
# --------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <title>TCS Stock Analytics</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: "Segoe UI", Arial, sans-serif;
            background: linear-gradient(
                135deg, #eef2ff, #f8fafc, #ecfeff
            );
            margin: 0;
            padding: 30px;
            color: #172033;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            background: linear-gradient(
                120deg, #101b36, #1d4ed8, #4338ca
            );
            padding: 30px;
            border-radius: 18px;
            margin-bottom: 26px;
            box-shadow: 0 10px 25px #1e3a8a20;
        }

        h1 {
            margin: 0 0 10px;
            font-size: 32px;
            color: #ffffff;
        }

        .subtitle {
            color: #dbeafe;
            font-size: 15px;
            line-height: 1.7;
        }

        .status {
            display: inline-block;
            margin-top: 17px;
            background: #ffffff20;
            border: 1px solid #ffffff40;
            color: white;
            padding: 7px 12px;
            border-radius: 20px;
            font-size: 12px;
        }

        .section-title {
            margin: 25px 0 16px;
            font-size: 21px;
            color: #172554;
        }

        .cards {
            display: grid;
            grid-template-columns: repeat(
                auto-fit, minmax(210px, 1fr)
            );
            gap: 20px;
        }

        .card {
            border-radius: 16px;
            padding: 23px;
            box-shadow: 0 6px 18px #0f172a0d;
            border: 1px solid white;
            transition: transform 0.2s, box-shadow 0.2s;
            min-height: 145px;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 25px #0f172a18;
        }

        .card:nth-child(1) {
            background: #e0f2fe;
            border-left: 5px solid #0284c7;
        }

        .card:nth-child(2) {
            background: #dcfce7;
            border-left: 5px solid #16a34a;
        }

        .card:nth-child(3) {
            background: #f3e8ff;
            border-left: 5px solid #9333ea;
        }

        .card:nth-child(4) {
            background: #ffedd5;
            border-left: 5px solid #ea580c;
        }

        .card:nth-child(5) {
            background: #ffe4e6;
            border-left: 5px solid #e11d48;
        }

        .card:nth-child(6) {
            background: #cffafe;
            border-left: 5px solid #0891b2;
        }

        .label {
            color: #475569;
            font-size: 14px;
            font-weight: 600;
            line-height: 1.5;
        }

        .value {
            font-size: 27px;
            font-weight: 750;
            margin-top: 14px;
            overflow-wrap: anywhere;
        }

        .card:nth-child(1) .value { color: #075985; }
        .card:nth-child(2) .value { color: #166534; }
        .card:nth-child(3) .value { color: #7e22ce; }
        .card:nth-child(4) .value { color: #9a3412; }
        .card:nth-child(5) .value { color: #9f1239; }
        .card:nth-child(6) .value { color: #155e75; }

        .small-text {
            color: #64748b;
            font-size: 12px;
            margin-top: 9px;
            line-height: 1.5;
        }

        .chart {
            background: white;
            margin-top: 28px;
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 6px 22px #0f172a0d;
        }

        .chart h2 {
            color: #172554;
            font-size: 21px;
            margin-top: 0;
        }

        .chart-description {
            color: #64748b;
            font-size: 14px;
            line-height: 1.7;
            margin-bottom: 20px;
        }

        .chart img {
            display: block;
            width: 100%;
            height: auto;
            border-radius: 10px;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 18px;
            background: white;
        }

        .comparison-table th {
            background: #1e3a8a;
            color: white;
            padding: 15px;
            text-align: left;
            font-size: 14px;
        }

        .comparison-table td {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
            font-size: 14px;
        }

        .comparison-table tbody tr:nth-child(even) {
            background: #f1f5f9;
        }

        .comparison-table tbody tr:hover {
            background: #dbeafe;
        }

        .result-note {
            background: #eff6ff;
            border-left: 4px solid #2563eb;
            color: #1e3a8a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 18px;
            line-height: 1.7;
            font-size: 14px;
        }

        .note {
            background: #e2e8f0;
            color: #475569;
            padding: 16px 18px;
            border-radius: 12px;
            font-size: 13px;
            line-height: 1.8;
            margin-top: 22px;
        }

        footer {
            text-align: center;
            color: #64748b;
            font-size: 12px;
            margin-top: 25px;
            padding: 10px;
        }

        @media (max-width: 600px) {
            body {
                padding: 14px;
            }

            header {
                padding: 22px;
            }

            h1 {
                font-size: 25px;
            }

            .cards {
                grid-template-columns: 1fr;
            }

            .card {
                padding: 19px;
            }

            .value {
                font-size: 24px;
            }

            .chart {
                padding: 15px;
            }

            .comparison-table th,
            .comparison-table td {
                padding: 10px 7px;
                font-size: 12px;
            }
        }
    </style>
</head>

<body>
<div class="container">

    <header>
        <h1>📈 TCS Stock Analytics</h1>

        <div class="subtitle">
            Machine Learning Model: Linear Regression
        </div>

        <div class="status">
            ● Dataset loaded | Model trained
        </div>
    </header>

    <h2 class="section-title">Stock Market Overview</h2>

    <div class="cards">

        <div class="card">
            <div class="label">Latest Available Date</div>
            <div class="value">{{ dashboard.latest_date }}</div>
            <div class="small-text">
                Most recent date in the dataset
            </div>
        </div>

        <div class="card">
            <div class="label">Latest Closing Price</div>
            <div class="value">
                ₹{{ "%.2f"|format(dashboard.latest_close) }}
            </div>
            <div class="small-text">
                Actual closing price in the dataset
            </div>
        </div>

        <div class="card">
            <div class="label">Estimated Next Close</div>
            <div class="value">
                ₹{{ "%.2f"|format(dashboard.estimated_next_close) }}
            </div>
            <div class="small-text">
                Model estimate, not a guarantee
            </div>
        </div>

        <div class="card">
            <div class="label">Test MAE</div>
            <div class="value">
                ₹{{ "%.2f"|format(dashboard.mae) }}
            </div>
            <div class="small-text">
                Mean Absolute Error — lower is better
            </div>
        </div>

        <div class="card">
            <div class="label">Test RMSE</div>
            <div class="value">
                ₹{{ "%.2f"|format(dashboard.rmse) }}
            </div>
            <div class="small-text">
                Root Mean Squared Error — lower is better
            </div>
        </div>

        <div class="card">
            <div class="label">Historical Records</div>
            <div class="value">{{ dashboard.total_records }}</div>
            <div class="small-text">
                Records in the original CSV
            </div>
        </div>

    </div>

    <!-- Historical chart -->

    <div class="chart">
        <h2>📊 Historical Stock Price Trend</h2>

        <div class="chart-description">
            TCS closing prices across the latest 100 available records.
        </div>

        <img
            src="data:image/png;base64,{{ dashboard.chart_image }}"
            alt="TCS historical closing price chart">
    </div>

    <!-- Actual vs predicted chart -->

    <div class="chart">
        <h2>📈 Actual vs Predicted Stock Prices</h2>

        <div class="chart-description">
            Actual closing prices are compared with Linear Regression
            predictions on the chronological test dataset.
        </div>

        <img
            src="data:image/png;base64,{{ dashboard.actual_predicted_chart_image }}"
            alt="Actual versus predicted TCS stock prices">
    </div>

    <!-- Model comparison -->

    <div class="chart">
        <h2>🔍 Machine Learning Model Comparison</h2>

        <div class="chart-description">
            Both models are evaluated on the same test data.
            Lower MAE and RMSE indicate smaller prediction errors.
        </div>

        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>MAE (₹)</th>
                    <th>RMSE (₹)</th>
                </tr>
            </thead>

            <tbody>
                {% for item in dashboard.model_comparison %}
                <tr>
                    <td>{{ item.name }}</td>
                    <td>{{ "%.2f"|format(item.mae) }}</td>
                    <td>{{ "%.2f"|format(item.rmse) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="result-note">
            <strong>Lower MAE model:</strong>
            {{ dashboard.better_model }}.

            This comparison identifies the model with the smaller
            MAE on this test set. Consider both MAE and RMSE when
            evaluating performance.
        </div>
    </div>

    <div class="note">
        <strong>About this project:</strong>
        This project uses historical TCS stock data and Linear
        Regression to estimate the next available trading day's
        closing price. The Baseline model predicts that the next
        closing price will equal the current closing price.

        <br><br>

        <strong>Disclaimer:</strong>
        This is an educational college project, not financial advice.
        Stock prices can change unpredictably, and model predictions
        are not guaranteed to be accurate.
    </div>

    <footer>
        TCS Stock Price Prediction |
        Python • Flask • Pandas • Scikit-learn • Matplotlib
    </footer>

</div>
</body>
</html>
"""


# --------------------------------------------------
# 9. FLASK ROUTE
# --------------------------------------------------

@app.route("/")
def home():
    try:
        dashboard = create_dashboard_data()

        return render_template_string(
            HTML_TEMPLATE,
            dashboard=dashboard
        )

    except Exception:
        app.logger.exception("Dashboard failed to load")

        return (
            "<h2>Unable to load dashboard</h2>"
            "<p>Check your dataset, Python packages, and terminal "
            "for the error details.</p>",
            500
        )


# --------------------------------------------------
# 10. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
