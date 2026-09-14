# Stock Screener Backtest — 10 Classic Screening Rules Tested Over 20 Years in Python

A Python backtesting engine that runs **10 classic stock-screening rules** — Low P/E,
High Dividend Yield, PEG < 1, Price-to-Book < 1.5, High ROE, Near 52-Week Low, Golden
Cross, RSI Oversold, 12-1 Momentum, and 10-Year Dividend Growth — against 20 years of
real historical price and fundamentals data, and ranks them by CAGR, max drawdown, and
Sharpe ratio.

If you've ever wondered whether "buy low P/E stocks" or "buy the Golden Cross" actually
outperforms over two decades, this script gives you a real, reproducible answer instead
of a hunch.

## What it does

- Pulls 20 years (2005–2025) of daily adjusted-close prices and full fundamentals for a
  fixed universe of 20 large-cap US stocks.
- Rebalances annually: on each rebalance date, every ticker is screened against a rule;
  tickers that pass are held equally-weighted until the next rebalance.
- Reports **CAGR** (compound annual growth rate), **max drawdown**, and **Sharpe ratio**
  for each of the 10 rules, so they can be compared head-to-head.

## Results (2005–2025, 20-stock universe)

| Rule | CAGR | Max Drawdown | Sharpe |
|---|---|---|---|
| High ROE | 12.5% | -21.7% | 0.86 |
| Near 52-Week Low | 12.4% | -22.0% | 0.92 |
| Golden Cross | 10.5% | -24.9% | 0.74 |
| 10-Year Dividend Growth | 10.4% | -18.9% | 0.83 |
| 12-1 Momentum | 9.3% | -26.4% | 0.61 |
| RSI Oversold | 8.7% | -23.5% | 0.31 |
| Low P/E | 8.7% | -22.9% | 0.56 |
| PEG < 1 | 7.0% | -25.4% | 0.40 |
| High Dividend Yield | 6.9% | -21.5% | 0.47 |
| Price-to-Book < 1.5 | 0.0% | 0.0% | N/A |

Full run output: [screening_rules_backtest_results.csv](screening_rules_backtest_results.csv)

## Known limitations (read before trusting the numbers)

- **Look-ahead bias on fundamental rules.** This starter version screens every rebalance
  date using EODHD's *current* fundamentals snapshot, not a point-in-time one — so 2010
  gets screened with today's financials. This affects Low P/E, High Dividend Yield,
  PEG < 1, Price-to-Book, High ROE, and Near 52-Week Low. Technical rules (Golden Cross,
  RSI Oversold, 12-1 Momentum) are unaffected, since they're computed from the actual
  historical price series. 10-Year Dividend Growth is also unaffected — it's checked
  against `SplitsDividends.NumberDividendsByYear`, a genuine per-year historical record.
- **Survivorship bias.** The 20-stock universe is fixed and hand-picked for
  reproducibility — it is *not* the historical S&P 500 constituent list for each year.
  Swap in a point-in-time constituent list if you have one.
- **Fixed rf rate.** Sharpe ratio uses a flat 2% risk-free rate for the entire 20-year
  window rather than the actual historical risk-free rate at each point.

## Setup

```bash
pip install -r requirements.txt
export EODHD_API_TOKEN="your_key_here"
python backtest_screening_rules.py
```

Results print to the console and are written to `screening_rules_backtest_results.csv`.

## Data source

All price and fundamentals data comes from the **[EODHD](https://eodhd.com/?via=kmg&ref1=Meneses&utm_source=medium&utm_medium=post&utm_campaign=stock-screening-rules-backtest-20-years-eodhd&utm_content=Meneses) API** — end-of-day
prices, fundamentals (P/E, ROE, dividend history, book value), and technical indicators
for 60+ global exchanges, all through one REST API with a generous free tier. If you want
to run backtests like this one yourself, [grab an EODHD API key](https://eodhd.com/?via=kmg&ref1=Meneses&utm_source=medium&utm_medium=post&utm_campaign=stock-screening-rules-backtest-20-years-eodhd&utm_content=Meneses).

## Adding your own screening rule

Every rule is a plain function `(ticker, fnd, prices, as_of=None) -> bool`:

```python
def rule_my_screen(ticker, fnd, prices, as_of=None):
    pe = fnd.get("Highlights", {}).get("PERatio")
    return pe is not None and 0 < float(pe) < 15
```

Add it to the `RULES` dict and it's automatically included in the next backtest run.
