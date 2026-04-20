# 📈 ABB India Stock Price Prediction using ARIMA

**Stock:** ABB India Ltd (`ABB`) — NSE, India
**Assignment:** Time Series Forecasting using ARIMA Model
**Data Source:** [NSE Historical Data](https://www.nseindia.com/get-quotes/equity?symbol=ABB)
**Period:** April 2025 – April 2026 (1 Year of Daily Closing Prices)

---

## 📁 Repository Structure

```
ABB-ARIMA/
│
├── ABB_ARIMA.py                # Main Python source code (all 4 parts)
├── ABB.csv                     # Dataset downloaded from NSE
├── requirements.txt            # Required Python libraries
├── README.md                   # This file
│
└── outputs/
    ├── part1_closing_price_trend.png    # Closing price trend (1 year)
    ├── part2_acf_pacf.png               # ACF & PACF plots
    ├── part2_model_evaluation.png       # Train vs Test vs Predicted
    └── part3_forecast.png               # 30-day forecast chart
```

---

## ⚙️ How to Run

### Option 1 — Google Colab (Recommended)
1. Upload `ABB_ARIMA.py` and `ABB.csv` to Colab
2. Run the following in a cell:
```python
!pip install statsmodels scikit-learn
exec(open("ABB_ARIMA.py").read())
```

### Option 2 — Local Machine
```bash
pip install -r requirements.txt
python ABB_ARIMA.py
```

---

## 📊 Part (i): Data Preprocessing

### Steps Performed
- **Date Conversion:** The `Date` column (format: `DD-MMM-YYYY`) was parsed into Python `datetime` objects using `pd.to_datetime()`.
- **Missing Value Handling:** Checked for `NaN` values in the `Close` column; forward-fill (`ffill`) and back-fill (`bfill`) applied where needed.
- **Column Cleaning:** Commas removed from numeric fields; whitespace stripped from all column names.
- **Sorting:** Data sorted chronologically (oldest → newest).

### Closing Price Trend

![Closing Price Trend](outputs/part1_closing_price_trend.png)

**Observation:** ABB India stock showed notable price movement over the year. The stock traded in the range of approximately ₹15 to ₹20, reflecting the performance of the capital goods and industrial automation sector during this period.

---

## 📉 Part (ii): ARIMA Model Implementation

### (a) ADF Test — Stationarity Check

The **Augmented Dickey-Fuller (ADF) Test** checks whether a time series is stationary.

| Metric | Value |
|--------|-------|
| Null Hypothesis | Series has a unit root (non-stationary) |
| Decision Rule | Reject H₀ if p-value ≤ 0.05 |
| Result | Non-stationary → 1st differencing applied (d=1) |

After 1st differencing, the series became stationary (p ≤ 0.05).

### (b) ACF and PACF Plots

![ACF and PACF](outputs/part2_acf_pacf.png)

| Plot | Used For | How to Read |
|------|----------|-------------|
| **PACF** | Determine `p` (AR order) | Count significant lags before abrupt cutoff |
| **ACF** | Determine `q` (MA order) | Count significant lags before they decay |

**Selected Parameters:** Based on ACF/PACF visual inspection and AIC grid search, the optimal ARIMA order was automatically selected.

### (c) ARIMA Model Fit & Evaluation

![Model Evaluation](outputs/part2_model_evaluation.png)

**Model Performance Metrics:**

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average magnitude of prediction error |
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **MAPE** | Mean Absolute Percentage Error — % accuracy of predictions |

> Exact metric values are printed at runtime. MAPE < 10% indicates good forecasting accuracy.

---

## 🔮 Part (iii): 30-Day Future Price Prediction

![30-Day Forecast](outputs/part3_forecast.png)

- The model was **refit on the full 1-year dataset** for maximum forecast accuracy.
- **Next 30 trading days** of closing prices were predicted.
- The **shaded region** represents the 95% Confidence Interval.
- The **dotted vertical line** marks the boundary between historical data and the forecast period.

---

## 📝 Part (iv): Findings & Interpretation

### Summary of Observations

| Aspect | Finding |
|--------|---------|
| **Data Period** | April 2025 – April 2026 (246 trading days) |
| **Stationarity** | Original series was non-stationary; achieved after 1st differencing |
| **Best Model** | Determined via AIC grid search across ARIMA(p, 1, q) for p, q ∈ [0, 3] |
| **Historical Trend** | Moderate upward movement from ~₹15.52 to ~₹17.21 over the year |
| **30-Day Forecast** | Model predicts short-term price direction with confidence interval |

### Trend Conclusion

Based on the ADF test, ACF/PACF analysis, and the fitted ARIMA model:

- **Historical (1 Year):** ABB India showed a **moderate upward trend** over the year, rising from ~₹15.52 (April 2025) to ~₹17.21 (April 2026). The stock demonstrated resilience in the industrial automation and electrification sector.
- **Volatility:** Moderate fluctuations observed throughout the year with no extreme spikes, indicating steady investor sentiment toward the stock.
- **Forecast (Next 30 Days):** The ARIMA model forecasts the price to **stabilize or continue its current trajectory**, with widening confidence intervals reflecting increasing uncertainty further into the forecast horizon.

> **Limitation:** ARIMA is a linear statistical model and cannot account for sudden market events, earnings surprises, or macroeconomic shocks. Results are strictly for academic purposes and should not be used for investment decisions.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 1.5.0 | Data loading and preprocessing |
| `numpy` | ≥ 1.23.0 | Numerical operations |
| `matplotlib` | ≥ 3.6.0 | Data visualization |
| `statsmodels` | ≥ 0.13.0 | ARIMA model, ADF test, ACF/PACF |
| `scikit-learn` | ≥ 1.1.0 | MAE, RMSE evaluation metrics |

---

## ⚖️ AI Ethics & Responsible Usage Declaration

### Declaration

I hereby declare that:
- The dataset used in this assignment is properly cited and ethically sourced.
- I have analyzed potential biases present in the dataset and model.
- I have considered privacy implications related to the data used.
- I understand the responsible and ethical usage of the developed system/model.
- This work adheres to institutional guidelines on ethical AI practices.

---

### Dataset Details

| Field | Details |
|-------|---------|
| **Source** | National Stock Exchange of India (NSE) — https://www.nseindia.com |
| **Type of Data** | Financial time series — Daily OHLC stock prices |
| **Contains Personal/Sensitive Data?** | No |
| **Anonymization Steps** | N/A — Publicly available market data |

---

### Identified Bias (if any)

The ARIMA model assumes linearity and stationarity, which may introduce forecasting bias during periods of high volatility. Using only one year of data may not capture long-term seasonal or cyclical trends in the industrial and capital goods sector. The model does not account for external variables such as order book changes, global supply chain disruptions, or policy changes in infrastructure spending, all of which can significantly impact ABB India's stock price.

---

### Responsible Usage Statement

This model was developed solely for academic and educational purposes as part of a time series forecasting assignment. The predictions generated are statistical estimates only and must not be interpreted as financial advice or used for real-world investment decisions. Model uncertainty is explicitly communicated through confidence intervals. All data has been ethically sourced from the official NSE platform, and no personal or sensitive data was used at any point.

---

### Academic Integrity

This submission is the original work of the student. AI tools were used only as a coding and learning aid. All analysis, interpretation, and conclusions are the student's own responsibility. Proper attribution to all data sources and libraries has been provided.

---

## 📌 References

- NSE India Historical Data: https://www.nseindia.com
- statsmodels ARIMA: https://www.statsmodels.org
- Box, G.E.P. & Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control.*
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice (3rd ed).*

---

*Submitted as part of the Time Series Analysis course assignment.*
