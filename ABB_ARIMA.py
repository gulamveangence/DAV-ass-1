# ============================================================
#  ABB Stock Analysis - ARIMA Model Assignment
#  NSE Stock : ABB India Ltd
#  Data      : Downloaded from NSE Historical Data (1 Year)
# ============================================================

# -------------------------------------------------------
# INSTALL REQUIRED LIBRARIES (run once)
# pip install statsmodels matplotlib pandas scikit-learn
# -------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os
os.makedirs("outputs", exist_ok=True)


# ============================================================
# PART (i) — DATA PREPROCESSING
# ============================================================

print("=" * 60)
print("PART (i): DATA PREPROCESSING")
print("=" * 60)

df = pd.read_csv("ABB.csv")
df.columns = df.columns.str.strip()

print("\nColumns detected:", list(df.columns))
print("\nFirst 5 rows (raw):")
print(df.head())

# (a) Convert Date column to proper datetime format
df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%b-%Y")
df = df.sort_values("Date").reset_index(drop=True)
df.set_index("Date", inplace=True)

print(f"\nDate column converted to datetime format.")
print(f"Date range   : {df.index[0].date()} to {df.index[-1].date()}")
print(f"Trading days : {len(df)}")

# Clean Close column
df["Close"] = df["Close"].astype(str).str.strip().str.replace(",", "").astype(float)

# (b) Handle missing values
print(f"\nMissing values before handling: {df['Close'].isna().sum()}")
df["Close"] = df["Close"].ffill().bfill()
print(f"Missing values after handling : {df['Close'].isna().sum()}")

close = df["Close"]

# (c) Visualize closing price trend over time
plt.figure(figsize=(14, 5))
plt.plot(close.index, close.values, color="#6a0dad", linewidth=1.5, label="Close Price")
plt.title("ABB India — Daily Closing Price (Past 1 Year, NSE Data)", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/part1_closing_price_trend.png", dpi=150)
plt.show()
print("\n[Saved] outputs/part1_closing_price_trend.png")


# ============================================================
# PART (ii) — ARIMA MODEL IMPLEMENTATION
# ============================================================

print("\n" + "=" * 60)
print("PART (ii): ARIMA MODEL IMPLEMENTATION")
print("=" * 60)

# (a) ADF Test
print("\n--- (a) Augmented Dickey-Fuller (ADF) Test ---")

def adf_test(series, label="Series"):
    result = adfuller(series.dropna())
    print(f"\nADF Test on: {label}")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Critical Values:")
    for key, val in result[4].items():
        print(f"    {key}: {val:.4f}")
    if result[1] <= 0.05:
        print("  >> STATIONARY (p <= 0.05, reject null hypothesis)")
    else:
        print("  >> NON-STATIONARY (p > 0.05, fail to reject null hypothesis)")
    return result[1] <= 0.05

is_stationary = adf_test(close, "Original Close Price")

d = 0
close_diff = close.copy()
if not is_stationary:
    close_diff = close.diff().dropna()
    d = 1
    print(f"\nApplying 1st order differencing (d=1)...")
    adf_test(close_diff, "1st Differenced Close Price")

# (b) ACF and PACF Plots
print(f"\n--- (b) ACF and PACF Plots (d={d}) ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(close_diff.dropna(),  lags=30, ax=axes[0], title="ACF — Differenced Close Price")
plot_pacf(close_diff.dropna(), lags=30, ax=axes[1], title="PACF — Differenced Close Price", method="ywm")
plt.suptitle("ACF & PACF Plots — ARIMA Parameter Selection (ABB India)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/part2_acf_pacf.png", dpi=150)
plt.show()
print("[Saved] outputs/part2_acf_pacf.png")

print(f"\n  p (AR order) -> read PACF: significant lags before cutoff")
print(f"  q (MA order) -> read ACF : significant lags before cutoff")
print(f"  d = {d}")

# (c) Fit ARIMA — AIC grid search
print(f"\n--- (c) Fitting ARIMA Model ---")
best_aic, best_order = np.inf, (1, d, 1)
print("Running AIC grid search for best (p, d, q)...")

for p in range(0, 4):
    for q in range(0, 4):
        try:
            m = ARIMA(close, order=(p, d, q)).fit()
            if m.aic < best_aic:
                best_aic, best_order = m.aic, (p, d, q)
        except Exception:
            continue

print(f"\nBest ARIMA Order : ARIMA{best_order}  (AIC = {best_aic:.2f})")

# Train / Test split (80/20)
train_size = int(len(close) * 0.8)
train, test = close.iloc[:train_size], close.iloc[train_size:]
print(f"Training samples : {len(train)}")
print(f"Testing samples  : {len(test)}")

fitted_model = ARIMA(train, order=best_order).fit()
print("\n--- ARIMA Model Summary ---")
print(fitted_model.summary())

predictions = pd.Series(
    fitted_model.forecast(steps=len(test)).values,
    index=test.index
)

mae  = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))
mape = np.mean(np.abs((test.values - predictions.values) / test.values)) * 100

print(f"\n--- Model Performance ---")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAPE : {mape:.2f}%")

plt.figure(figsize=(14, 5))
plt.plot(train.index, train.values, label="Training Data",       color="#1f77b4", linewidth=1.2)
plt.plot(test.index,  test.values,  label="Actual (Test)",       color="#2ca02c", linewidth=1.5)
plt.plot(predictions.index, predictions.values,
         label=f"Predicted ARIMA{best_order}", color="#d62728", linewidth=1.5, linestyle="--")
plt.title(f"ABB India — ARIMA{best_order}: Train / Test / Predicted", fontsize=13, fontweight="bold")
plt.xlabel("Date"); plt.ylabel("Price (INR)"); plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("outputs/part2_model_evaluation.png", dpi=150)
plt.show()
print("[Saved] outputs/part2_model_evaluation.png")


# ============================================================
# PART (iii) — FUTURE PRICE PREDICTION (Next 30 Days)
# ============================================================

print("\n" + "=" * 60)
print("PART (iii): FUTURE PRICE PREDICTION — Next 30 Days")
print("=" * 60)

final_fitted    = ARIMA(close, order=best_order).fit()
forecast_result = final_fitted.get_forecast(steps=30)
forecast_mean   = forecast_result.predicted_mean
forecast_ci     = forecast_result.conf_int(alpha=0.05)

last_date       = close.index[-1]
forecast_dates  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
forecast_series = pd.Series(forecast_mean.values, index=forecast_dates)
ci_lower        = pd.Series(forecast_ci.iloc[:, 0].values, index=forecast_dates)
ci_upper        = pd.Series(forecast_ci.iloc[:, 1].values, index=forecast_dates)

print(f"\n{'Date':<15} {'Forecasted Price (INR)':>22}")
print("-" * 38)
for date, price in zip(forecast_dates, forecast_series):
    print(f"{str(date.date()):<15} Rs. {price:>17.2f}")

plt.figure(figsize=(14, 6))
plt.plot(close.index[-90:], close.values[-90:],
         label="Historical (Last 90 days)", color="#6a0dad", linewidth=1.5)
plt.plot(forecast_series.index, forecast_series.values,
         label="30-Day Forecast", color="#d62728", linewidth=2, linestyle="--")
plt.fill_between(forecast_dates, ci_lower, ci_upper,
                 color="#d62728", alpha=0.15, label="95% Confidence Interval")
plt.axvline(x=last_date, color="gray", linestyle=":", linewidth=1.5, label="Forecast Start")
plt.title(f"ABB India — ARIMA{best_order}: 30-Day Price Forecast", fontsize=13, fontweight="bold")
plt.xlabel("Date"); plt.ylabel("Price (INR)"); plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("outputs/part3_forecast.png", dpi=150)
plt.show()
print("[Saved] outputs/part3_forecast.png")


# ============================================================
# PART (iv) — INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("PART (iv): INTERPRETATION OF RESULTS")
print("=" * 60)

start_price     = float(close.iloc[0])
end_price       = float(close.iloc[-1])
forecast_end    = float(forecast_series.iloc[-1])
overall_change  = ((end_price - start_price) / start_price) * 100
forecast_change = ((forecast_end - end_price) / end_price) * 100

trend_hist = "UPWARD" if overall_change  >  5 else "DOWNWARD" if overall_change  < -5 else "STABLE"
trend_fore = "UPWARD" if forecast_change >  2 else "DOWNWARD" if forecast_change < -2 else "STABLE"

print(f"""
FINAL RESULTS SUMMARY
======================
Stock            : ABB India Ltd — NSE
ARIMA Order      : {best_order}
AIC              : {best_aic:.2f}

Historical (1 Year)
  Start : Rs. {start_price:.2f}  |  End : Rs. {end_price:.2f}
  Change: {overall_change:+.2f}%  |  Trend: {trend_hist}

Model Performance
  MAE : {mae:.4f}  |  RMSE : {rmse:.4f}  |  MAPE : {mape:.2f}%

30-Day Forecast
  End Price : Rs. {forecast_end:.2f}  |  Change: {forecast_change:+.2f}%
  Trend     : {trend_fore}

Conclusion: The stock is expected to {'RISE' if forecast_change > 2 else 'DECLINE' if forecast_change < -2 else 'REMAIN STABLE'}
over the next 30 trading days. ({trend_fore} TREND)
""")

print("=" * 60)
print("All 4 parts complete. Charts saved in outputs/ folder.")
print("=" * 60)
