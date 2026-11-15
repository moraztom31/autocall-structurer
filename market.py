"""
Market data for the note builder.

What is available free and what is not, stated plainly:

  Spot levels      available, delayed, good enough to build a term sheet
  Realised vol     computable from history
  Implied vol      NOT available - and it is the input that matters most

A Phoenix is short volatility through its barriers. A one-point error in
implied volatility moves the structuring margin by roughly one point of
nominal, which is more than the margin floor itself. Realised volatility
is a starting point, never a substitute: it is backward-looking and
typically sits below implied for equity indices, which flatters margin.

So the builder offers both, and labels which one is in use.
"""

from dataclasses import dataclass

import numpy as np

# Indices with a liquid listed-option market, where an implied vol the
# user supplies is meaningful. Decrement indices are not quoted on free
# data sources, so the builder applies a synthetic decrement instead.
INDICES = {
    "^STOXX50E": "EURO STOXX 50",
    "^GDAXI": "DAX 40",
    "^FCHI": "CAC 40",
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq 100",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
    "^SSMI": "SMI 20",
}

DEFAULT_VOL = 0.18          # used when no history and no user input
DEFAULT_RATE = 0.030        # flat continuously compounded fallback


@dataclass
class Quote:
    """A spot level and a realised-volatility estimate for one index."""

    ticker: str
    name: str
    spot: float
    realised_vol: float | None
    as_of: str
    history: object = None       # pandas Series of closes, or None
    source: str = "live"         # "live" or "manual"
    error: str | None = None

    @property
    def is_live(self):
        return self.source == "live" and self.error is None


def realised_volatility(closes, window=252):
    """
    Annualised standard deviation of daily log returns.

    A backward-looking estimate. It carries no volatility risk premium
    and no forward view, so it generally understates the implied level
    an issuer would actually trade on.
    """
    closes = np.asarray(closes, dtype=float)
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if closes.size < 30:
        return None
    returns = np.diff(np.log(closes))[-window:]
    if returns.size < 20:
        return None
    return float(returns.std(ddof=1) * np.sqrt(252))


def fetch_quote(ticker, period="2y"):
    """
    Pull spot and price history for one index.

    Never raises: any failure - missing package, no network, unknown
    ticker, empty frame - comes back as a Quote carrying an error, so the
    interface can fall back to manual entry instead of breaking.
    """
    name = INDICES.get(ticker, ticker)

    try:
        import yfinance as yf
    except ImportError:
        return Quote(ticker, name, float("nan"), None, "", None,
                     "manual", "yfinance is not installed.")

    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            return Quote(ticker, name, float("nan"), None, "", None,
                         "manual", f"No price history returned for {ticker}.")

        closes = hist["Close"].dropna()
        if closes.empty:
            return Quote(ticker, name, float("nan"), None, "", None,
                         "manual", f"Empty price series for {ticker}.")

        return Quote(
            ticker=ticker,
            name=name,
            spot=float(closes.iloc[-1]),
            realised_vol=realised_volatility(closes.values),
            as_of=str(closes.index[-1].date()),
            history=closes,
            source="live",
        )
    except Exception as exc:                       # noqa: BLE001
        return Quote(ticker, name, float("nan"), None, "", None,
                     "manual", f"Could not reach the data provider: {exc}")


def manual_quote(spot, vol, name="Custom underlying"):
    """A quote assembled from values typed in by the user."""
    return Quote(ticker="CUSTOM", name=name, spot=float(spot),
                 realised_vol=float(vol), as_of="user input",
                 history=None, source="manual")
