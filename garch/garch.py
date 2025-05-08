import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

# ───────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ───────────────────────────────────────────────────────────────────────────────
CSV_PATH            = '../data/btcusd_1-min_data.csv'

# 1) How to aggregate & forecast: 'D' = daily, 'h' = hourly, '30min' = 30 min, etc.
TIME_SCALE          = 'h'

# 2) Horizon in units of TIME_SCALE to forecast ahead (1 => next day, 3 => 3 days ahead etc.)
PREDICTION_HORIZON  = 1

# 3) GARCH spec
GARCH_P, GARCH_Q    = 1, 1
MEAN_MODEL          = 'Zero'
DIST                = 'StudentsT'
# ───────────────────────────────────────────────────────────────────────────────

# 1) Load & resample
df = pd.read_csv(CSV_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Timestamp', inplace=True)

# Resample close price at chosen frequency, take last tick
price = df['Close'].resample(TIME_SCALE).last().dropna()

# Compute log-returns at that frequency
returns = np.log(price / price.shift(1)).dropna() * 100

# 2) Quick stationarity & ARCH checking (optional)
adf_stat, adf_p, *_ = adfuller(returns)
print(f"ADF on {TIME_SCALE}-log-returns: p={adf_p:.3g}")

arch_stat, arch_p, *_ = het_arch(returns)
print(f"ARCH test on {TIME_SCALE}-log-returns: p={arch_p:.3g}")

# 3) Train/Test split (80/20)
split = int(len(returns) * 0.8)
train, test = returns[:split], returns[split:]

# 4) Fit GARCH
model     = arch_model(returns, vol='Garch', p=GARCH_P, q=GARCH_Q,
                       mean=MEAN_MODEL, dist=DIST)
res       = model.fit(disp='off')
print(res.summary())

# 5) Backtest across the test‐set
print("Backtesting...")
r = model.fit(last_obs=train.index[-1], disp='off')
print("model fitted")
fc = r.forecast(start=train.index[-1], horizon=PREDICTION_HORIZON)
print("forcast done")
print(fc.residual_variance)
preds = fc.residual_variance.iloc[1:, PREDICTION_HORIZON - 1] # We start at the second row because the first row is the last train date.
preds = preds.shift(PREDICTION_HORIZON) # Align with the date being forcasted
preds = preds.reindex(test.index).dropna()
print(f"First date: {test.index[0]}, last date: {test.index[-1]}")

test_aligned = test.reindex(preds.index)

test_aligned = test_aligned ** 2

# Simple metrics
errors = test_aligned - preds
print("MSE:",        np.mean(errors**2))
print("MAE:",        np.mean(np.abs(errors)))
print("QLIKE:",      np.mean(np.log(preds) + test_aligned/preds))

# ───────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ───────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Actual vs Predicted Variance
ax1.plot(test_aligned.index, test_aligned, label='Actual Variance', alpha=0.7, color='blue', marker='o')
ax1.plot(test_aligned.index, preds, label='Predicted Variance', alpha=0.8, color='red', linewidth=2, marker='o')
ax1.set_title(f'GARCH({GARCH_P},{GARCH_Q}) Actual vs Predicted Variance - {TIME_SCALE} Time Scale')
ax1.set_ylabel('Variance')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Forecast Errors
ax2.plot(test_aligned.index, errors, label='Prediction Error', alpha=0.7, color='green')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, marker='o')
ax2.set_title('Prediction Errors (Actual - Predicted)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Error')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualization: Scatter plot of actual vs predicted
plt.figure(figsize=(8, 8))
plt.scatter(preds, test_aligned, alpha=0.6)
plt.plot([0, max(max(preds), max(test_aligned))], [0, max(max(preds), max(test_aligned))], 'r--', label='Perfect Prediction')
plt.xlabel('Predicted Variance')
plt.ylabel('Actual Variance')
plt.title(f'Actual vs Predicted Variance - {TIME_SCALE} Time Scale')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Summary statistics for the visualization
print("\nVisualization Summary:")
print(f"Correlation between actual and predicted: {np.corrcoef(preds, test_aligned)[0,1]:.3f}")
print(f"Number of test points: {len(test_aligned)}")
print(f"Average predicted variance: {np.mean(preds):.4f}")
print(f"Average actual variance: {np.mean(test_aligned):.4f}")

