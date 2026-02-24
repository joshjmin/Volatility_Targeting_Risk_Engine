"""
Volatility-Targeting Backtester
================================
Dynamically adjusts SPY exposure based on EWMA volatility to maintain
a constant 10% target volatility. Compares against a static 60/40 portfolio.

Core Features:
- Recursive EWMA volatility (λ=0.94)
- Dynamic position sizing
- Proper 1-day lag (no look-ahead bias)
- Full comparative analysis vs 60/40 benchmark
- Comprehensive visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

# Assets
RISKY_ASSET = 'QQQ'
SAFE_ASSET = 'IEF'

# Risk Parameters
TARGET_VOLATILITY = 0.10       # 10% annualized target
LAMBDA_DECAY = 0.94            # EWMA decay factor
TRADING_DAYS_PER_YEAR = 252    # For annualization

# Backtest Period
YEARS_OF_DATA = 5

# Benchmark
BENCHMARK_RISKY_WEIGHT = 0.60  # 60/40 portfolio
BENCHMARK_SAFE_WEIGHT = 0.40

# ============================================================================
# DATA ACQUISITION
# ============================================================================

print("=" * 80)
print("VOLATILITY-TARGETING BACKTESTER")
print("=" * 80)
print(f"\nDownloading {YEARS_OF_DATA} years of data...")
print(f"  Risky Asset: {RISKY_ASSET}")
print(f"  Safe Asset:  {SAFE_ASSET}")

# Set date range
end_date = datetime.now()
start_date = end_date - timedelta(days=YEARS_OF_DATA * 365)

# Download both assets
tickers = [RISKY_ASSET, SAFE_ASSET]
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# Extract closing prices
prices = data['Close']

print(f"\nDownloaded {len(prices)} trading days")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# ============================================================================
# CALCULATE RETURNS
# ============================================================================

print("\nCalculating log returns...")

# Log returns for both assets
returns = np.log(prices / prices.shift(1)).dropna()

print(f"Calculated returns for {len(returns)} periods")
print(f"\nReturn Statistics (Annualized):")
print(f"  {RISKY_ASSET}: {float(returns[RISKY_ASSET].mean())*TRADING_DAYS_PER_YEAR*100:6.2f}% return, "
      f"{float(returns[RISKY_ASSET].std())*np.sqrt(TRADING_DAYS_PER_YEAR)*100:6.2f}% volatility")
print(f"  {SAFE_ASSET}:  {float(returns[SAFE_ASSET].mean())*TRADING_DAYS_PER_YEAR*100:6.2f}% return, "
      f"{float(returns[SAFE_ASSET].std())*np.sqrt(TRADING_DAYS_PER_YEAR)*100:6.2f}% volatility")

# ============================================================================
# CALCULATE EWMA VOLATILITY FOR SPY
# ============================================================================

print(f"\nCalculating EWMA volatility for {RISKY_ASSET} (λ={LAMBDA_DECAY})...")

# Initialize
n = len(returns)
variance = np.zeros(n)
variance[0] = returns[RISKY_ASSET].values[:20].var()

# Recursive EWMA: σ²_t = λ × σ²_{t-1} + (1-λ) × r²_{t-1}
for t in range(1, n):
    variance[t] = (LAMBDA_DECAY * variance[t-1] + 
                   (1 - LAMBDA_DECAY) * (returns[RISKY_ASSET].values[t-1] ** 2))

# Convert to annualized volatility
volatility_daily = np.sqrt(variance)
volatility_annual = volatility_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

# Store as Series
spy_volatility = pd.Series(volatility_annual, index=returns.index)

print("EWMA volatility calculated")
print(f"  Current volatility: {spy_volatility.iloc[-1]*100:.2f}%")
print(f"  Average volatility: {spy_volatility.mean()*100:.2f}%")
print(f"  Min/Max: {spy_volatility.min()*100:.2f}% / {spy_volatility.max()*100:.2f}%")

# ============================================================================
# CALCULATE VOLATILITY-TARGETED WEIGHTS
# ============================================================================

print(f"\nCalculating volatility-targeted weights (target={TARGET_VOLATILITY*100}%)...")

# Weight for SPY: min(1.0, target_vol / current_vol)
raw_spy_weight = TARGET_VOLATILITY / spy_volatility
spy_weight = np.minimum(raw_spy_weight, 1.0)  # Cap at 100%

# Weight for IEF: remainder
ief_weight = 1.0 - spy_weight

# Store as DataFrame
weights = pd.DataFrame({
    RISKY_ASSET: spy_weight,
    SAFE_ASSET: ief_weight
}, index=returns.index)

print("Weights calculated")
print(f"\nCurrent allocation:")
print(f"  {RISKY_ASSET}: {weights[RISKY_ASSET].iloc[-1]*100:5.1f}%")
print(f"  {SAFE_ASSET}:  {weights[SAFE_ASSET].iloc[-1]*100:5.1f}%")
print(f"\nAverage Allocation (Full Period):")
print(f"  {RISKY_ASSET}: {weights[RISKY_ASSET].mean()*100:5.1f}%")
print(f"  {SAFE_ASSET}:  {weights[SAFE_ASSET].mean()*100:5.1f}%")

# ============================================================================
# BACKTEST THE STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("BACKTESTING")
print("=" * 80)

print("\nApplying 1-day lag to weights (avoiding look-ahead bias)...")

# CRITICAL: Shift weights by 1 day
lagged_weights = weights.shift(1)

# Calculate strategy returns
strategy_returns = (lagged_weights * returns).sum(axis=1)

# Remove NaN from first period
strategy_returns = strategy_returns.dropna()

print(f"Strategy backtest complete ({len(strategy_returns)} periods)")

# ============================================================================
# CALCULATE BENCHMARK (60/40 PORTFOLIO)
# ============================================================================

print("\nCalculating 60/40 benchmark...")

# Static 60/40 allocation
benchmark_returns = (BENCHMARK_RISKY_WEIGHT * returns[RISKY_ASSET] + 
                     BENCHMARK_SAFE_WEIGHT * returns[SAFE_ASSET])

# Align to strategy dates
benchmark_returns = benchmark_returns.loc[strategy_returns.index]

print("Benchmark calculated")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

def calculate_metrics(returns, name):
    """Calculate comprehensive performance metrics."""
    # Total return
    total_return = float((1 + returns).prod() - 1)
    
    # Annualized return (CAGR)
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    cagr = float((1 + total_return) ** (1 / n_years) - 1)
    
    # Annualized volatility
    vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    # Sharpe Ratio
    sharpe = cagr / vol if vol > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())
    
    # VaR (95%)
    var_95 = float(returns.quantile(0.05))
    
    # Win Rate
    win_rate = float((returns > 0).sum() / len(returns))
    
    print(f"\n{name}:")
    print(f"  Total Return:        {total_return*100:8.2f}%")
    print(f"  Annualized Return:   {cagr*100:8.2f}%")
    print(f"  Annualized Vol:      {vol*100:8.2f}%")
    print(f"  Sharpe Ratio:        {sharpe:8.3f}")
    print(f"  Maximum Drawdown:    {abs(max_dd)*100:8.2f}%")
    print(f"  VaR (95%, daily):    {abs(var_95)*100:8.3f}%")
    print(f"  Win Rate:            {win_rate*100:8.1f}%")
    
    return {
        'returns': returns,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'var_95': var_95,
        'win_rate': win_rate,
        'cumulative': cumulative,
        'drawdown': drawdown
    }

# Calculate metrics
strategy_metrics = calculate_metrics(strategy_returns, "Volatility-Targeting Strategy")
benchmark_metrics = calculate_metrics(benchmark_returns, "60/40 Benchmark")

# Show improvement
print("\n" + "-" * 80)
print("IMPROVEMENT OVER BENCHMARK:")
print(f"  Sharpe Ratio:        {(strategy_metrics['sharpe'] - benchmark_metrics['sharpe']):+8.3f}")
print(f"  Max Drawdown:        {(abs(strategy_metrics['max_drawdown']) - abs(benchmark_metrics['max_drawdown']))*100:+8.2f}%")
print(f"  Annualized Return:   {(strategy_metrics['cagr'] - benchmark_metrics['cagr'])*100:+8.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING CHARTS")
print("=" * 80)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# ============================================================================
# CHART 1: CUMULATIVE RETURNS
# ============================================================================

ax1 = axes[0]

strategy_cum = strategy_metrics['cumulative']
benchmark_cum = benchmark_metrics['cumulative']

ax1.plot(strategy_cum.index, strategy_cum.values, 
         label='Volatility-Targeting Strategy', linewidth=2.5, color='#2E86AB')
ax1.plot(benchmark_cum.index, benchmark_cum.values, 
         label='60/40 Benchmark', linewidth=2.5, color='#A23B72', linestyle='--')

ax1.set_title('Cumulative Returns: Strategy vs Benchmark', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Return ($)', fontsize=11)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(strategy_cum.index[0], strategy_cum.index[-1])

# Add final values as text
final_strat = strategy_cum.iloc[-1]
final_bench = benchmark_cum.iloc[-1]
ax1.text(0.02, 0.95, 
         f'Final Value:\nStrategy: ${final_strat:.2f}\nBenchmark: ${final_bench:.2f}',
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         verticalalignment='top')

# ============================================================================
# CHART 2: ROLLING DRAWDOWN
# ============================================================================

ax2 = axes[1]

strategy_dd = strategy_metrics['drawdown'] * 100
benchmark_dd = benchmark_metrics['drawdown'] * 100

ax2.fill_between(strategy_dd.index, strategy_dd.values, 0, 
                 alpha=0.4, color='#2E86AB', label='Volatility-Targeting Strategy')
ax2.fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                 alpha=0.4, color='#A23B72', label='60/40 Benchmark')

ax2.set_title('Drawdown Profile: Risk Protection During Market Stress', fontsize=14, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.legend(loc='lower left', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(strategy_dd.index[0], strategy_dd.index[-1])

# Add max drawdown annotations
ax2.text(0.02, 0.05, 
         f'Max Drawdown:\nStrategy: {abs(strategy_metrics["max_drawdown"])*100:.1f}%\n'
         f'Benchmark: {abs(benchmark_metrics["max_drawdown"])*100:.1f}%',
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         verticalalignment='bottom')

# ============================================================================
# CHART 3: DYNAMIC ALLOCATION
# ============================================================================

ax3 = axes[2]

# Use lagged weights (what was actually traded)
actual_weights = weights.shift(1).dropna() * 100

ax3.fill_between(actual_weights.index, 0, actual_weights[RISKY_ASSET].values,
                 alpha=0.6, color='#2E86AB', label=f'{RISKY_ASSET} (Stocks)')
ax3.fill_between(actual_weights.index, actual_weights[RISKY_ASSET].values, 100,
                 alpha=0.6, color='#06A77D', label=f'{SAFE_ASSET} (Bonds)')

ax3.set_title('Dynamic Asset Allocation (Volatility-Targeting)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Allocation (%)', fontsize=11)
ax3.set_xlabel('Date', fontsize=11)
ax3.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(actual_weights.index[0], actual_weights.index[-1])
ax3.set_ylim(0, 100)

# Add target volatility line (overlaid)
ax3_twin = ax3.twinx()
ax3_twin.plot(spy_volatility.index, spy_volatility.values * 100, 
              color='#F18F01', linewidth=1.5, alpha=0.7, label='SPY Volatility')
ax3_twin.axhline(y=TARGET_VOLATILITY * 100, color='red', linestyle='--', 
                 linewidth=2, label='Target Vol (10%)', alpha=0.7)
ax3_twin.set_ylabel('Volatility (%)', fontsize=11)
ax3_twin.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3_twin.set_ylim(0, max(spy_volatility.max() * 100 * 1.1, 30))

# ============================================================================
# FINALIZE
# ============================================================================

plt.tight_layout()

output_path = 'volatility_targeting_backtest.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nChart saved to: {output_path}")

# ============================================================================
# AI EXPLANATION FOR VOLATILITY SPIKES
# ============================================================================

def generate_ai_vol_spike_explanation(asset_name, vol_series, return_series):
    """Generate a short explanation of a recent volatility spike for the given asset using OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        print("\nopenai package not installed. Run `pip install openai` to enable volatility explanations.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nOPENAI_API_KEY not set. Set it in your environment to enable volatility explanations.")
        return None

    client = OpenAI(api_key=api_key)

    # Focus on the last ~30 calendar days to give the model recent context
    end_date = vol_series.index[-1]
    start_date = end_date - pd.Timedelta(days=30)

    vol_window = vol_series.loc[start_date:end_date]
    ret_window = return_series.loc[start_date:end_date]

    if vol_window.empty or ret_window.empty:
        print("\nNot enough recent data to describe a volatility spike.")
        return None

    stats_summary = (
        f"Asset: {asset_name}\n"
        f"Window: {start_date.date()} to {end_date.date()}\n"
        f"Start volatility: {vol_window.iloc[0]*100:.2f}%\n"
        f"Peak volatility:  {vol_window.max()*100:.2f}%\n"
        f"End volatility:   {vol_window.iloc[-1]*100:.2f}%\n"
        f"Average volatility in window: {vol_window.mean()*100:.2f}%\n"
        f"Total {asset_name} return over window: {(1+ret_window).prod()-1:+.2%}\n"
        f"Worst daily return: {ret_window.min():+.2%}\n"
        f"Best daily return:  {ret_window.max():+.2%}"
    )

    system_prompt = (
        "You are a concise, professional macro and market analyst. "
        "You are given a short summary of a recent volatility spike in a traded asset. "
        "Using your knowledge of common market drivers (macro data, central bank policy, earnings, "
        "geopolitics, liquidity shocks, etc.) and the provided stats, explain in 2–3 short paragraphs "
        "what could plausibly have caused such a spike, and what type of environment it is consistent with. "
        "If the spike is mild, say that explicitly and frame it as a normal regime shift or noise. "
        "Do NOT give trading advice or position recommendations."
    )

    user_prompt = (
        f"Here is a summary of the recent behavior of {asset_name} volatility and returns:\n\n"
        f"{stats_summary}\n\n"
        "Based on this, explain the likely macro or market context behind this volatility pattern. "
        "Keep the explanation focused, non-technical, and under 250 words."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.4,
        )
    except Exception as e:
        print(f"\nError while calling OpenAI API: {e}")
        return None

    content = response.choices[0].message.content if response and response.choices else None
    if not content:
        print("\nNo content returned from OpenAI.")
        return None

    return content.strip()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("BACKTEST SUMMARY")
print("=" * 80)

print(f"\nStrategy: Volatility-Targeting (Target: {TARGET_VOLATILITY*100}%)")
print(f"Benchmark: Static 60/40 Portfolio")
print(f"Period: {strategy_returns.index[0].date()} to {strategy_returns.index[-1].date()}")
print(f"Trading Days: {len(strategy_returns)}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Insight 1: Risk-Adjusted Returns
sharpe_improvement = strategy_metrics['sharpe'] - benchmark_metrics['sharpe']
if sharpe_improvement > 0.1:
    print(f"\n✓ SUPERIOR RISK-ADJUSTED RETURNS")
    print(f"  Strategy Sharpe ({strategy_metrics['sharpe']:.3f}) beats benchmark ({benchmark_metrics['sharpe']:.3f})")
    print(f"  Improvement: +{sharpe_improvement:.3f}")
else:
    print("\nRisk-adjusted returns similar to benchmark")
    print("  Strategy and benchmark have comparable Sharpe ratios")

# Insight 2: Drawdown Protection
dd_improvement = abs(benchmark_metrics['max_drawdown']) - abs(strategy_metrics['max_drawdown'])
if dd_improvement > 0.05:
    print("\nShallower drawdowns than benchmark")
    print(f"  Strategy drawdown ({abs(strategy_metrics['max_drawdown'])*100:.1f}%) is {dd_improvement*100:.1f}% better than benchmark")
    print("  Better capital preservation during market stress")
elif dd_improvement < -0.05:
    print("\nDeeper drawdowns than benchmark")
    print(f"  Strategy experienced {abs(dd_improvement)*100:.1f}% worse drawdown than benchmark")
else:
    print("\nDrawdown risk similar to benchmark")

# Insight 3: Volatility Targeting Effectiveness
realized_vol = strategy_metrics['volatility']
vol_accuracy = abs(realized_vol - TARGET_VOLATILITY) / TARGET_VOLATILITY
if vol_accuracy < 0.15:
    print("\nVolatility close to target")
    print(f"  Realized vol ({realized_vol*100:.2f}%) is within 15% of target ({TARGET_VOLATILITY*100}%)")
    print(f"  Deviation: {(realized_vol - TARGET_VOLATILITY)*100:+.2f}%")
else:
    print("\nVolatility meaningfully different from target")
    print(f"  Realized vol ({realized_vol*100:.2f}%) differs from target ({TARGET_VOLATILITY*100}%)")
    print("  Consider adjusting the decay parameter")

# Insight 4: Current Market Regime
current_vol = spy_volatility.iloc[-1]
avg_vol = spy_volatility.mean()

print("\nCurrent market state:")
print(f"  {RISKY_ASSET} Volatility: {current_vol*100:.2f}% (avg: {avg_vol*100:.2f}%)")
print(f"  Current {RISKY_ASSET} Weight: {weights[RISKY_ASSET].iloc[-1]*100:.1f}%")

if current_vol > avg_vol * 1.3:
    print("  Volatility is elevated relative to its recent history")
elif current_vol < avg_vol * 0.7:
    print("  Volatility is low relative to its recent history")
else:
    print("  Volatility is near its recent average")

# ============================================================================
# Volatility spike explanation
# ============================================================================

print("\n" + "=" * 80)
print("Volatility spike explanation")
print("=" * 80)

if current_vol > avg_vol * 1.3:
    ai_explanation = generate_ai_vol_spike_explanation(RISKY_ASSET, spy_volatility, returns[RISKY_ASSET])
    if ai_explanation:
        print("\n" + ai_explanation)
    else:
        print("\nNo explanation available.")
else:
    print("\nNo major recent volatility spike detected: AI explanation not needed.")

print("\n" + "=" * 80)
print("Backtest complete")
print("=" * 80)
print(f"\nView detailed charts: {output_path}")