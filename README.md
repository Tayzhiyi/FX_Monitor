# FX Portfolio Management Dashboard

**Dashboard:** https://59c9a187-9151-428e-b71e-14b27e470a71.plotly.app/

A Dash application for monitoring P&L and risk for a synthetic FX spot portfolio. Positions are read from `data/portfolio.csv`, while current and historical FX prices are obtained from Yahoo Finance via `yfinance`.

## Running locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the dashboard:

```powershell
python main.py
```

Then open:

```text
http://127.0.0.1:8050/
```

The sample portfolio can be regenerated with:

```powershell
python -m src.create_portfolio
```

The application was developed and tested locally using **Python 3.9.6**. The deployed Plotly Cloud version uses **Python 3.10** for hosting compatibility.

---

## Methodology and major assumptions

### Portfolio and P&L

Each trade contains a trade date, FX pair, side, base-currency notional and entry price. `LONG` means long the base currency / short the quote currency, while `SHORT` means the reverse. The current implementation supports FX pairs where USD is either the base or quote currency.

Let `d = +1` for LONG and `-1` for SHORT, `N` be base notional, `K` entry price and `S` spot.

For USD-quote pairs such as AUDUSD:

```text
P&L = d × N × (S - K)
```

For USD-base pairs such as USDJPY:

```text
P&L = d × N × (1 - K / S)
```

Inception P&L is current MTM relative to entry. Daily P&L is current MTM less MTM at the previous completed UTC daily FX close.

Headline P&L percentages use **gross initial USD notional** as an exposure denominator. USD-base notionals are already in USD; USD-quote notionals are converted using the trade entry price. Long and short notionals are summed on a gross basis. This is an exposure-normalised P&L measure, not a true portfolio/NAV return.

The valuation assumes the original currency legs remain outstanding and excludes financing, carry, transaction costs and intervening cash flows.

### Market data and stale values

Current prices are sourced from Yahoo Finance. The application first attempts to use 1-minute intraday data and falls back to the latest daily observation if necessary.

Intraday quotes are labelled:

- `LIVE`: up to 30 minutes old
- `DELAYED`: more than 30 and up to 120 minutes old
- `STALE`: more than 120 minutes old
- `UNAVAILABLE`: failed validation

A `STALE` quote can still be used if it is no more than **24 hours old** and passes the other validation checks. This is intentional because some Yahoo FX series update less frequently. The dashboard displays quote status and age so the user can judge data quality.

Quotes are also rejected if they are invalid, materially ahead of the valuation timestamp, not newer than the previous-close baseline, or more than 20% away from the previous completed close.

Historical daily bars are treated as UTC-day bars. A generic Monday-Friday business-day calendar is used; pair-specific holidays are not modelled.

### VaR and risk

The dashboard uses a **1-day delta-normal variance-covariance VaR**. The default is 95% confidence using 252 valid aligned daily FX returns, with alternative confidence levels and lookbacks available in the UI.

Simple FX returns are used:

```text
r(t) = S(t) / S(t-1) - 1
```

Missing prices are not forward-filled, and returns spanning a missing weekday are excluded.

Each trade is converted into a signed USD sensitivity to a proportional FX move:

```text
USD-quote pair: x = d × N × S
USD-base pair:  x = d × N × K / S
```

so approximately:

```text
ΔP&L ≈ x × FX return
```

For multiple trades in the same FX pair, signed USD sensitivities are **netted first**. Portfolio VaR is then calculated across the unique FX risk factors using their covariance matrix:

```text
Portfolio volatility = sqrt(x' Σ x)
Portfolio VaR        = z × Portfolio volatility
```

A 1-day 95% VaR of USD 30,000 means that, under the model assumptions, the one-day loss is modelled to exceed about USD 30,000 on roughly 5% of days. VaR is **not** a maximum possible loss.

The dashboard also calculates:

- **Standalone VaR:** risk of a trade or net pair in isolation
- **Marginal VaR:** local change in portfolio VaR for an additional unit of risk
- **Component VaR:** allocation of portfolio VaR back to trades/pairs; negative values indicate hedging/diversification
- **Incremental VaR:** change in portfolio VaR if a trade or pair is removed; this is not additive
- **Expected Shortfall:** expected loss conditional on exceeding the VaR threshold
- **Within-pair Netting Benefit:** reduction from offsetting trades in the same FX pair
- **Cross-pair Diversification Benefit:** further reduction from correlations between different FX pairs

The stress test complements VaR by performing exact revaluation under ±1% and ±2% coherent USD shocks.

### P&L percentages and performance ratios

The Daily and Inception P&L percentages shown in the headline cards are calculated as:

```text
P&L % = USD P&L / gross initial USD notional
```

Gross initial USD notional is the sum of absolute USD-equivalent trade notionals. For USD-base pairs, the base notional is already in USD; for USD-quote pairs, base notional is converted using the trade's entry price. Long and short trades are included on a gross basis rather than netted.

This percentage is an **exposure-normalised P&L measure, not a true portfolio return**, because the sample portfolio does not contain a NAV or invested-capital series.

For the same reason, the displayed Sharpe and Sortino ratios are **P&L-based rather than return-based**. They use completed full-portfolio daily USD P&L observations:

```text
Sharpe = mean(daily P&L) / sample standard deviation(daily P&L) × sqrt(252)
```

The benchmark is USD 0 daily P&L. Sortino uses the same numerator and annualisation, but divides by downside deviation relative to a USD 0 target. These metrics are therefore descriptive risk-adjusted P&L measures rather than conventional NAV-return Sharpe/Sortino ratios, and should be interpreted cautiously given the short sample history.

---

## Limitations / known issues

- The deployed Plotly Cloud version can load and refresh more slowly than the local app. This is mainly due to hosted-app startup / network latency and the external Yahoo Finance requests made when market data is refreshed; local calculations such as changing VaR confidence or lookback are typically faster.
- Yahoo Finance is not an institutional real-time FX data source; quotes may be delayed, stale or unavailable.
- Only FX pairs containing USD are currently supported. Crosses such as EURGBP would require an additional USD conversion path.
- Pair orientation is assumed to be consistent. For example, `USDJPY` and `JPYUSD` are not automatically normalised into one risk factor.
- VaR assumes normally distributed returns, zero expected one-day return and locally linear P&L. It may understate tail risk or large nonlinear moves.
- USD-base P&L is nonlinear in spot, so delta-normal VaR is an approximation.
- Volatility and correlation estimates are historical and may not represent future stressed regimes.
- A generic Monday-Friday calendar is used instead of currency-specific holiday calendars.
- The displayed P&L percentage is based on gross initial USD notional, not NAV or invested capital.


---

## Improvements with more time

I would prioritise:

1. replacing Yahoo Finance with institutional-quality market data;
2. supporting non-USD crosses and canonical pair orientation;
3. adding FX forwards / NDFs, carry and settlement mechanics;
4. adding currency-specific holiday calendars;
5. adding **cross-asset risk-equivalent mapping** by returns against benchmark macro factors such as DXY or Treasury, then using the estimated factor sensitivities and covariance structure to express the portfolio's risk in equivalent benchmark exposures;
6. adding Historical Simulation or Monte Carlo VaR as a nonlinear benchmark to delta-normal VaR;
7. adding VaR backtesting and configurable risk-limit alerts; and
8. incorporating portfolio NAV / capital so performance metrics can be calculated using true portfolio returns.

---

## AI usage

AI tools were used as development assistance, particularly for documentation, improving code readability and presentation (including comments, explanatory wording, formatting and indentation), as well as debugging and code review.
