# Volatility-Targeting Risk Engine

A quantitative backtesting framework that dynamically manages portfolio risk by targeting a constant annualized volatility. The strategy adjusts equity exposure in real time using Exponentially Weighted Moving Average (EWMA) volatility estimates, and benchmarks performance against a static 60/40 portfolio.

---

## What It Does

Rather than holding a fixed allocation, this engine continuously re-sizes equity exposure so that the portfolio's *realized* volatility tracks a user-defined target (default: 10% annualized). When markets become turbulent, the engine reduces risk automatically. When volatility is low, it scales up — all without requiring a human in the loop.

---

## Core Methodology

### EWMA Volatility Estimation (RiskMetrics Model)
Volatility is estimated using the recursive J.P. Morgan RiskMetrics formula:

```
σ²_t = λ · σ²_{t-1} + (1 - λ) · r²_{t-1}
```

- **λ = 0.94** (industry-standard decay factor, as used by RiskMetrics)
- Weights recent returns exponentially more than older ones
- Responds quickly to volatility spikes without overreacting to noise
- Initialized using the rolling variance of the first 20 observations

### Dynamic Position Sizing
The risky asset weight at each time step is:

```
w_t = min(1.0, σ_target / σ_t)
```

- Scales down equity when volatility rises above target
- Caps exposure at 100% (no leverage)
- Remainder is allocated to the safe asset (IEF — intermediate Treasuries)

### Look-Ahead Bias Prevention
Weights are **lagged by exactly one trading day** before being applied to returns. Today's volatility estimate informs tomorrow's trade — never the same day.

---

## Assets

Any pair of tickers supported by `yfinance` can be used — just update `RISKY_ASSET` and `SAFE_ASSET` in the config block. The defaults are:

| Role | Default Ticker | Description |
|------|---------------|-------------|
| Risky Asset | SPY | S&P 500 ETF (broad U.S. equity) |
| Safe Asset | IEF | 7–10 Year Treasury Bond ETF |

Examples of other valid combinations: QQQ/TLT, BTC-USD/SHY, GLD/SHV.

---

## Performance Analysis

The backtester computes and compares the following metrics for both the strategy and the 60/40 benchmark:

| Metric | Description |
|--------|-------------|
| CAGR | Compound Annual Growth Rate |
| Annualized Volatility | Realized standard deviation of log returns × √252 |
| Sharpe Ratio | CAGR divided by realized volatility (risk-free rate = 0) |
| Maximum Drawdown | Largest peak-to-trough decline |
| VaR (95%) | 5th percentile of daily return distribution |
| Win Rate | Fraction of positive-return trading days |

---

## Visualizations

The engine outputs a three-panel chart (`volatility_targeting_backtest.png`):

1. **Cumulative Returns** — Strategy vs. 60/40 benchmark over the full period
2. **Drawdown Profile** — Rolling drawdown comparison; highlights capital preservation during stress
3. **Dynamic Allocation** — Day-by-day equity/bond split, overlaid with realized EWMA volatility vs. the 10% target

---

## Configuration

All parameters are centralized at the top of the script:

```python
TARGET_VOLATILITY = 0.10        # Annualized vol target
LAMBDA_DECAY = 0.94             # EWMA smoothing factor
YEARS_OF_DATA = 5               # Backtest window
BENCHMARK_RISKY_WEIGHT = 0.60   # 60/40 benchmark equity weight
RISKY_ASSET = 'SPY'
SAFE_ASSET = 'IEF'
```

---

## Setup & Usage

```bash
# Install dependencies
pip install numpy pandas matplotlib yfinance

# Run the backtest
python volatility_targeting_backtest.py
```

Output:
- Printed performance metrics and regime analysis in the terminal
- `volatility_targeting_backtest.png` — full chart panel

---

## Optional: AI Volatility Spike Explanation

If `OPENAI_API_KEY` is set in the environment and a recent volatility spike is detected (current vol > 130% of trailing average), the engine calls GPT-4o-mini to generate a concise macro explanation of the regime shift. This is purely informational and produces no trading signals.

```bash
export OPENAI_API_KEY=your_key_here
python volatility_targeting_backtest.py
```

---

## Key Design Decisions

- **Log returns** are used throughout for time-additivity and symmetry
- **No leverage** — equity weight is hard-capped at 100%
- **No transaction costs** modeled — a realistic extension for live deployment
- **EWMA over rolling window** — avoids the "ghost effect" of equal-weighted rolling vol dropping off a volatile day abruptly

---

## Potential Extensions

- Add transaction cost modeling with turnover constraints
- Incorporate a second volatility asset (e.g., VIX futures) for regime overlay
- Extend to multi-asset portfolios with covariance-based targeting
- Add walk-forward optimization for λ and target volatility parameters
- Connect to a broker API (Alpaca, IBKR) for live execution
