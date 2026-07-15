# Arbitrage-Free Volatility Surface

Tayte Choudhury

Install dependencies and reproduce the labeled synthetic validation:

```sh
python -m pip install -r requirements.txt
python surface.py --demo --out results
python -m unittest test_surface.py
```

To analyze a real synchronized snapshot:

```sh
python surface.py --csv snapshot.csv --out market-results
```

Required columns: timestamp, spot, expiry_years, strike, type, bid, ask.
type is C or P. Prices are in dollars; expiry_years is a positive year fraction.
Require one timestamp and spot, unique (expiry,strike,type) contracts and both
calls and puts per expiry for forward/discount inference. Supply European-style
quotes appropriate to Black-76; American exercise effects require separate treatment.

Five liquidity screens are counted sequentially. The relative spread limit is 30%,
the minimum maturity is seven days and the minimum coverage is six strikes.
The code fits five-parameter raw SVI with positivity, Lee wing slopes, Durrleman
density and adjacent-expiry calendar constraints. Density and calendar checks use
2,000 log-forward-moneyness points between -1 and 1. Outside this finite grid and
between sampled points, absence of arbitrage is not guaranteed.

The bundled example has 4,320 generated quotes over 18 expiries. It is not the
4,180-quote SPY snapshot described in the resume; that original dataset was absent.
Do not treat this demonstration as evidence of historical market fit or performance.
No exercise-style correction, dividends model or extrapolation engine is included.

Methodology: Gatheral & Jacquier, https://arxiv.org/abs/1204.0646

public/ contains an interactive presentation and diagnostics for static hosting.
Python code and input quotes are not served by that configuration.


Portfolio: https://t8c4.com/projects/volatility-surface.html

Repository: https://github.com/t8dahgr8/tayte-volatility-surface


The complete original folder layout, public presentation and Render configuration are included in project-bundle.zip. Extract that archive before following instructions referring to public/.
