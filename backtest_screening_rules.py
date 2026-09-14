"""
backtest_screening_rules.py

Backtests 10 classic stock-screening rules over a 20-year window using
EODHD's historical price, fundamentals, and technical-indicator APIs.

Run this in Claude Code:

    export EODHD_API_TOKEN=your_key_here
    pip install pandas numpy requests
    python backtest_screening_rules.py

LIMITATION (read before trusting the numbers):
This starter version screens each rebalance date using EODHD's *current*
fundamentals snapshot, not a point-in-time one. That introduces look-ahead
bias: you're technically screening 2010 with 2026 financials. For a rule
like "Golden Cross" or "RSI Oversold" this doesn't matter, since technical
values are pulled from the actual historical price series. For fundamental
rules (P/E, ROE, dividend yield, PEG, P/B) AND "Near 52-Week Low" (which
also reads a live snapshot field, `Technicals.52WeekLow`, not a
point-in-time one), treat the results as directional, not exact. To fix it
properly, pull the historical `Earnings.History` and `Financials` arrays
from the fundamentals response and compute trailing metrics as of each
rebalance date instead of using the latest `Highlights`/`Technicals` block.
That's noted inline below where it applies.

"10-Year Dividend Growth" is the exception: `SplitsDividends.NumberDividendsByYear`
is a real per-year historical record, so it's filtered to years on or before
each rebalance date and checked for an actual consecutive streak (verified
live: AAPL's 1996-2011 dividend suspension shows up correctly as a gap).
"""

import os
import time

import numpy as np
import pandas as pd
import requests

API_TOKEN = os.environ.get("EODHD_API_TOKEN", "YOUR_API_KEY")
BASE_URL = "https://eodhd.com/api"

START_DATE = "2005-01-01"
END_DATE = "2025-01-01"

# Fixed universe for reproducibility. This is NOT the historical S&P 500
# constituent list, so results carry survivorship bias. Swap in a
# point-in-time constituent list if you have one.
UNIVERSE = [
    "AAPL.US", "MSFT.US", "JNJ.US", "PG.US", "KO.US", "PEP.US", "XOM.US",
    "CVX.US", "JPM.US", "WMT.US", "HD.US", "MRK.US", "ABT.US", "MMM.US",
    "CAT.US", "IBM.US", "T.US", "VZ.US", "PFE.US", "INTC.US",
]


# ---- Data access -----------------------------------------------------------

def get_eod_prices(ticker, start=START_DATE, end=END_DATE):
    """Daily split/dividend-adjusted close."""
    url = f"{BASE_URL}/eod/{ticker}"
    params = {"api_token": API_TOKEN, "from": start, "to": end, "period": "d", "fmt": "json"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["adjusted_close"].astype(float)


def get_fundamentals(ticker):
    url = f"{BASE_URL}/fundamentals/{ticker}"
    params = {"api_token": API_TOKEN, "fmt": "json"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def get_latest_indicator(ticker, function, period=14):
    url = f"{BASE_URL}/technical/{ticker}"
    params = {"api_token": API_TOKEN, "function": function, "period": period, "fmt": "json"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    return data[-1][function] if data else None


# ---- Screening rules ---------------------------------------------------
# Each rule returns True/False for one ticker at the current rebalance pass.

def rule_low_pe(ticker, fnd, prices, sector_median_pe=20, as_of=None):
    pe = fnd.get("Highlights", {}).get("PERatio")
    return pe is not None and 0 < float(pe) < sector_median_pe


def rule_high_dividend_yield(ticker, fnd, prices, threshold=0.04, as_of=None):
    yld = fnd.get("Highlights", {}).get("DividendYield")
    return yld is not None and float(yld) >= threshold


def rule_peg_under_1(ticker, fnd, prices, as_of=None):
    peg = fnd.get("Highlights", {}).get("PEGRatio")
    return peg is not None and 0 < float(peg) < 1


def rule_low_price_to_book(ticker, fnd, prices, threshold=1.5, as_of=None):
    pb = fnd.get("Valuation", {}).get("PriceBookMRQ")
    return pb is not None and 0 < float(pb) < threshold


def rule_high_roe(ticker, fnd, prices, threshold=0.15, as_of=None):
    roe = fnd.get("Highlights", {}).get("ReturnOnEquityTTM")
    return roe is not None and float(roe) >= threshold


def rule_near_52w_low(ticker, fnd, prices, band=0.10, as_of=None):
    tech = fnd.get("Technicals", {})
    low = tech.get("52WeekLow")
    if low is None or prices.empty:
        return False
    latest_price = prices.iloc[-1]
    return latest_price <= float(low) * (1 + band)


def rule_golden_cross(ticker, fnd, prices, as_of=None):
    if len(prices) < 200:
        return False
    sma50 = prices.tail(50).mean()
    sma200 = prices.tail(200).mean()
    return sma50 > sma200


def rule_rsi_oversold(ticker, fnd, prices, threshold=30, as_of=None):
    if len(prices) < 15:
        return False
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0).tail(14).mean()
    loss = -delta.clip(upper=0).tail(14).mean()
    if loss == 0:
        return False
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi < threshold


def rule_momentum_12_1(ticker, fnd, prices, as_of=None):
    if len(prices) < 260:
        return False
    twelve_m_ago = prices.iloc[-252]
    one_m_ago = prices.iloc[-21]
    return (one_m_ago / twelve_m_ago - 1) > 0


def rule_dividend_growth_streak(ticker, fnd, prices, years=10, as_of=None):
    """True if the company paid a dividend in `years` straight calendar
    years ending on or before `as_of`.

    NumberDividendsByYear only contains years where Count > 0 (confirmed
    live: AAPL's 1996-2011 suspension is simply absent from the dict), so
    counting *distinct* years without checking consecutiveness or the
    rebalance date is wrong - it made this rule pass for all 20 tickers on
    every single rebalance date in the backtest, i.e. it was a no-op that
    just bought the whole universe every year.
    """
    raw = fnd.get("SplitsDividends", {}).get("NumberDividendsByYear")
    if not isinstance(raw, dict):
        return False
    paid_years = sorted({int(v["Year"]) for v in raw.values() if v.get("Count", 0) > 0})
    if as_of is not None:
        paid_years = [y for y in paid_years if y <= as_of.year]
    if not paid_years:
        return False
    streak = 1
    for i in range(len(paid_years) - 1, 0, -1):
        if paid_years[i] - paid_years[i - 1] == 1:
            streak += 1
        else:
            break
    return streak >= years


RULES = {
    "Low P/E": rule_low_pe,
    "High Dividend Yield": rule_high_dividend_yield,
    "PEG < 1": rule_peg_under_1,
    "Price-to-Book < 1.5": rule_low_price_to_book,
    "High ROE": rule_high_roe,
    "Near 52-Week Low": rule_near_52w_low,
    "Golden Cross": rule_golden_cross,
    "RSI Oversold": rule_rsi_oversold,
    "12-1 Momentum": rule_momentum_12_1,
    "10-Year Dividend Growth": rule_dividend_growth_streak,
}


# ---- Backtest engine -----------------------------------------------------

def annual_rebalance_dates(start, end):
    return pd.date_range(start=start, end=end, freq="YS")


def cagr(equity_curve):
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def sharpe_ratio(period_returns, rf=0.02):
    if len(period_returns) < 2:
        return float("nan")
    excess = pd.Series(period_returns) - rf
    std = excess.std()
    # Exact `== 0` misses floating-point noise: for a rule that never picks
    # any stock, period_returns is a constant [0.0, 0.0, ...], but std() on
    # that constant still comes out ~1e-18 instead of exactly 0 (confirmed
    # live), so mean/std blows up to something like -5.6e+15 instead of
    # being caught as "no variance, no sharpe to compute."
    if np.isclose(std, 0, atol=1e-9):
        return float("nan")
    return excess.mean() / std


def backtest_rule(rule_name, rule_fn, universe, price_data, fundamentals_cache):
    dates = annual_rebalance_dates(START_DATE, END_DATE)
    equity = [1.0]
    equity_dates = [dates[0]]
    period_returns = []

    for i in range(len(dates) - 1):
        period_start, period_end = dates[i], dates[i + 1]

        passing = []
        for ticker in universe:
            prices_to_date = price_data[ticker][price_data[ticker].index <= period_start]
            fnd = fundamentals_cache[ticker]
            try:
                if rule_fn(ticker, fnd, prices_to_date, as_of=period_start):
                    passing.append(ticker)
            except Exception:
                continue

        if not passing:
            equity.append(equity[-1])
            equity_dates.append(period_end)
            period_returns.append(0.0)
            continue

        returns_this_period = []
        for ticker in passing:
            p = price_data[ticker]
            p_period = p[(p.index >= period_start) & (p.index <= period_end)]
            if len(p_period) < 2:
                continue
            returns_this_period.append(p_period.iloc[-1] / p_period.iloc[0] - 1)

        avg_return = float(np.mean(returns_this_period)) if returns_this_period else 0.0
        equity.append(equity[-1] * (1 + avg_return))
        equity_dates.append(period_end)
        period_returns.append(avg_return)

    curve = pd.Series(equity, index=equity_dates)
    return {
        "rule": rule_name,
        "cagr": cagr(curve),
        "max_drawdown": max_drawdown(curve),
        "sharpe": sharpe_ratio(period_returns),
    }


def main():
    print(f"Fetching {len(UNIVERSE)} tickers of price history and fundamentals...")
    price_data = {}
    fundamentals_cache = {}
    for ticker in UNIVERSE:
        price_data[ticker] = get_eod_prices(ticker)
        fundamentals_cache[ticker] = get_fundamentals(ticker)
        time.sleep(0.2)  # basic rate-limit courtesy

    results = []
    for name, fn in RULES.items():
        print(f"Backtesting: {name}")
        results.append(backtest_rule(name, fn, UNIVERSE, price_data, fundamentals_cache))

    df = pd.DataFrame(results).sort_values("cagr", ascending=False)
    df["cagr"] = (df["cagr"] * 100).round(1)
    df["max_drawdown"] = (df["max_drawdown"] * 100).round(1)
    df["sharpe"] = df["sharpe"].round(2)
    print(df.to_string(index=False))
    df.to_csv("screening_rules_backtest_results.csv", index=False)


if __name__ == "__main__":
    main()
