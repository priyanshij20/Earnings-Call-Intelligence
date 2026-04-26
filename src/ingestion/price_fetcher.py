"""
Fetches historical stock price data from yfinance for all target tickers,
and computes post-earnings returns at 1, 3, and 7 day windows.
"""

import re
import json
import yaml
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm


def fetch_prices(tickers: list[str], start: str, end: str, output_dir: str):
    """Download OHLCV data for each ticker and save as CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ticker in tqdm(tickers, desc="Fetching stock prices"):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"  No data for {ticker}")
                continue
            df.to_csv(out / f"{ticker}.csv")
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    print(f"Saved price data to {output_dir}")


def compute_returns(transcript_dir: str, market_dir: str, windows: list[int]) -> pd.DataFrame:
    """
    For each parsed transcript, compute stock returns over given windows after the call date.
    Returns a DataFrame indexed by (ticker, quarter).
    """
    records = []
    market_path = Path(market_dir)
    transcript_path = Path(transcript_dir)

    for json_file in transcript_path.glob("*.json"):
        with open(json_file) as f:
            t = json.load(f)

        ticker = t['ticker']
        date_str = t['date']
        quarter = t['quarter']

        price_file = market_path / f"{ticker}.csv"
        if not price_file.exists():
            continue

        try:
            prices = pd.read_csv(price_file, index_col=0, skiprows=[1, 2])
            prices.index = pd.to_datetime(prices.index, errors='coerce')
            prices = prices[prices.index.notna()]

            # Normalize date string (may be list if JSON loaded weirdly)
            if isinstance(date_str, list):
                date_str = date_str[0] if date_str else ""
            if re.match(r'\d{4}-\d{2}-\d{2}', str(date_str)):
                call_date = pd.Timestamp(date_str)
            else:
                parts = [p.strip() for p in date_str.split(',')]
                # Try abbreviated month first, then full month name
                for fmt in ('%b %d, %Y', '%B %d, %Y'):
                    try:
                        call_date = pd.Timestamp(datetime.strptime(f"{parts[0]}, {parts[1]}", fmt))
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Cannot parse date: {date_str}")
            trading_days = prices.index[prices.index >= call_date]

            if len(trading_days) == 0:
                continue

            day0 = trading_days[0]
            day0_price = prices.loc[day0, 'Close']

            row = {"ticker": ticker, "quarter": quarter, "date": date_str, "day0_price": day0_price}

            for w in windows:
                if len(trading_days) > w:
                    day_w_price = prices.loc[trading_days[w], 'Close']
                    row[f"return_{w}d"] = (day_w_price - day0_price) / day0_price
                else:
                    row[f"return_{w}d"] = None

            records.append(row)

        except Exception as e:
            print(f"  Return calc failed {ticker} {quarter}: {e}")

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from src.config import load as load_cfg
    cfg = load_cfg()

    fetch_prices(
        cfg['tickers'],
        cfg['date_range']['start'],
        cfg['date_range']['end'],
        cfg['paths']['market']
    )

    returns_df = compute_returns(
        cfg['paths']['processed'],
        cfg['paths']['market'],
        cfg['stock_return_windows']
    )

    out = Path(cfg['paths']['features'])
    out.mkdir(parents=True, exist_ok=True)
    returns_df.to_csv(out / "returns.csv", index=False)
    print(f"Saved returns for {len(returns_df)} transcripts")
    print(returns_df.head())
