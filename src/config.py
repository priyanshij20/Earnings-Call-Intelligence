"""Config loader that normalizes settings.yaml into a flat, consistent dict."""

import yaml


def load(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    tickers = [
        (item["ticker"] if isinstance(item, dict) else item)
        for sector in raw["companies"].values()
        for item in sector
    ]

    paths = raw.get("paths", {})
    models = raw.get("models", {})

    return {
        "tickers": tickers,
        "date_range": raw.get("date_range", {}),
        "paths": {
            "raw_pkl":   paths.get("raw_data", "data/raw") + "/motley-fool-data.pkl"
                         if not paths.get("raw_data", "").endswith(".pkl")
                         else paths["raw_data"],
            "processed": paths.get("processed_data", paths.get("processed", "data/processed")),
            "market":    paths.get("market_data",    paths.get("market",    "data/market")),
            "features":  paths.get("features", "data/features"),
        },
        "models": {
            "finbert_sentiment":    (models.get("finbert", {}).get("model_name")
                                     or models.get("finbert_sentiment", "ProsusAI/finbert")),
            "sentence_transformer": models.get("sentence_transformer", "all-MiniLM-L6-v2"),
        },
        "stock_return_windows": (raw.get("returns", {}).get("windows")
                                  or raw.get("stock_return_windows", [1, 3, 7])),
        "language_shift_threshold": raw.get("language_shift_threshold", 0.85),
    }
