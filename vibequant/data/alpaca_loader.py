"""
Alpaca Data Loader Module
Fetches and caches historical bar data from Alpaca Markets API.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from dotenv import load_dotenv


class AlpacaDataLoader:
    """
    Handles all data fetching operations from Alpaca API.
    Features:
    - Multi-timeframe bar data (1Min, 5Min, 15Min, 1Hour, 1Day)
    - Intelligent caching to minimize API calls
    - Universe filtering by liquidity and asset type
    - Automatic data validation and cleaning
    """

    TIMEFRAME_MAP = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
        "1Week": TimeFrame(1, TimeFrameUnit.Week),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Alpaca Data Loader.

        Args:
            api_key: Alpaca API key (defaults to env var)
            secret_key: Alpaca secret key (defaults to env var)
            base_url: Alpaca base URL (defaults to env var)
            cache_dir: Directory to store cached data
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("ALPACA_PAPER_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_PAPER_SECRET_KEY")
        self.base_url = base_url or os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found. "
                "Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY in .env"
            )

        # Initialize clients
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(self.api_key, self.secret_key)

        # Setup cache directory
        self.cache_dir = Path(cache_dir or "./data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for universe data
        self._universe_cache: Optional[pd.DataFrame] = None
        self._universe_cache_time: Optional[datetime] = None

    def _get_cache_key(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> str:
        """Generate a unique cache key for the data request."""
        key_str = f"{sorted(symbols)}_{timeframe}_{start.date()}_{end.date()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        meta_file = self.cache_dir / f"{cache_key}_meta.json"

        if cache_file.exists() and meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)

            cache_time = datetime.fromisoformat(meta["cached_at"])
            # Cache is valid for 1 hour for intraday, 24 hours for daily
            max_age = timedelta(hours=1 if "Min" in meta["timeframe"] else 24)

            if datetime.now() - cache_time < max_age:
                return pd.read_parquet(cache_file)

        return None

    def _save_to_cache(
        self,
        data: pd.DataFrame,
        cache_key: str,
        timeframe: str,
        symbols: List[str],
    ) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        meta_file = self.cache_dir / f"{cache_key}_meta.json"

        data.to_parquet(cache_file)

        meta = {
            "cached_at": datetime.now().isoformat(),
            "timeframe": timeframe,
            "symbols": symbols,
            "rows": len(data),
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f)

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        lookback_days: int = 252,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical bar data for given symbols.

        Args:
            symbols: List of ticker symbols
            timeframe: One of "1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day", "1Week"
            start: Start datetime (defaults to lookback_days ago)
            end: End datetime (defaults to now)
            lookback_days: Number of trading days to look back if start not specified
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap, trade_count
            MultiIndex: (symbol, timestamp)
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid options: {list(self.TIMEFRAME_MAP.keys())}"
            )

        end = end or datetime.now()
        start = start or (end - timedelta(days=lookback_days))

        # Check cache
        cache_key = self._get_cache_key(symbols, timeframe, start, end)
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                print(f"[DataLoader] Loaded {len(cached_data)} bars from cache")
                return cached_data

        # Fetch from API
        print(f"[DataLoader] Fetching {timeframe} bars for {len(symbols)} symbols...")

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self.TIMEFRAME_MAP[timeframe],
            start=start,
            end=end,
        )

        bars = self.data_client.get_stock_bars(request)

        # Convert to DataFrame
        if not bars.data:
            print("[DataLoader] No data returned from API")
            return pd.DataFrame()

        records = []
        for symbol, symbol_bars in bars.data.items():
            for bar in symbol_bars:
                records.append({
                    "symbol": symbol,
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "vwap": float(bar.vwap) if bar.vwap else None,
                    "trade_count": int(bar.trade_count) if bar.trade_count else None,
                })

        df = pd.DataFrame(records)

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index(["symbol", "timestamp"]).sort_index()

        # Cache the data
        if use_cache:
            self._save_to_cache(df, cache_key, timeframe, symbols)
            print(f"[DataLoader] Cached {len(df)} bars")

        return df

    def get_liquid_universe(
        self,
        min_price: float = 5.0,
        max_price: float = 10000.0,
        min_volume: int = 500_000,
        include_etfs: bool = True,
        max_symbols: int = 500,
        refresh: bool = False,
    ) -> List[str]:
        """
        Get a universe of liquid, tradeable symbols.

        Args:
            min_price: Minimum stock price filter
            max_price: Maximum stock price filter
            min_volume: Minimum average daily volume
            include_etfs: Whether to include ETFs
            max_symbols: Maximum number of symbols to return
            refresh: Force refresh of cached universe

        Returns:
            List of ticker symbols meeting criteria
        """
        cache_file = self.cache_dir / "universe.json"

        # Check cache (valid for 24 hours)
        if not refresh and cache_file.exists():
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            cache_time = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cache_time < timedelta(hours=24):
                return cache_data["symbols"][:max_symbols]

        print("[DataLoader] Building liquid universe...")

        # Get all tradeable assets
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = self.trading_client.get_all_assets(request)

        # Filter for tradeable, non-OTC assets
        tradeable = [
            a for a in assets
            if a.tradable and not a.exchange == "OTC"
        ]

        symbols = [a.symbol for a in tradeable]
        print(f"[DataLoader] Found {len(symbols)} tradeable assets")

        # Get recent price/volume data to filter
        if len(symbols) > 0:
            # Sample recent data (last 20 days)
            end = datetime.now()
            start = end - timedelta(days=30)

            # Batch process to avoid API limits
            batch_size = 200
            all_stats = []

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Day,
                        start=start,
                        end=end,
                    )
                    bars = self.data_client.get_stock_bars(request)

                    for symbol, symbol_bars in bars.data.items():
                        if len(symbol_bars) >= 10:  # Need at least 10 days
                            closes = [float(b.close) for b in symbol_bars]
                            volumes = [int(b.volume) for b in symbol_bars]
                            avg_price = sum(closes) / len(closes)
                            avg_volume = sum(volumes) / len(volumes)

                            if (min_price <= avg_price <= max_price and
                                avg_volume >= min_volume):
                                all_stats.append({
                                    "symbol": symbol,
                                    "avg_price": avg_price,
                                    "avg_volume": avg_volume,
                                })
                except Exception as e:
                    print(f"[DataLoader] Batch error: {e}")
                    continue

            # Sort by volume (most liquid first)
            all_stats.sort(key=lambda x: x["avg_volume"], reverse=True)
            filtered_symbols = [s["symbol"] for s in all_stats[:max_symbols]]

            # Cache the results
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "symbols": filtered_symbols,
                "criteria": {
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_volume": min_volume,
                },
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            print(f"[DataLoader] Universe: {len(filtered_symbols)} liquid symbols")
            return filtered_symbols

        return []

    def get_popular_etfs(self) -> List[str]:
        """Get a list of popular, liquid ETFs for testing."""
        return [
            # Broad Market
            "SPY", "QQQ", "IWM", "DIA", "VTI",
            # Sector ETFs
            "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE",
            # International
            "EFA", "EEM", "VEU", "IEMG",
            # Bonds
            "TLT", "IEF", "LQD", "HYG", "BND",
            # Commodities
            "GLD", "SLV", "USO", "UNG",
            # Volatility
            "VXX", "UVXY",
            # Leveraged (for research)
            "TQQQ", "SQQQ", "SPXL", "SPXS",
        ]

    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 component symbols (cached list)."""
        # Common S&P 500 stocks - a representative sample
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK.B", "UNH", "XOM",
            "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
            "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
            "DHR", "NEE", "VZ", "ADBE", "TXN", "PM", "CMCSA", "NKE", "RTX", "NFLX",
            "BMY", "HON", "UNP", "ORCL", "QCOM", "COP", "LOW", "UPS", "MS", "SPGI",
            "IBM", "INTC", "CAT", "BA", "GE", "SBUX", "DE", "INTU", "AMD", "AMGN",
            "GS", "BLK", "MDLZ", "ISRG", "ADP", "GILD", "ADI", "SYK", "C", "REGN",
            "T", "LMT", "PLD", "BKNG", "VRTX", "TMUS", "ZTS", "CI", "SCHW", "CVS",
        ]

    def get_bars_safe(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        lookback_days: int = 252,
        batch_size: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch bars with graceful error handling for invalid/delisted symbols.
        Processes symbols in batches and skips ones that cause errors.

        Args:
            symbols: List of ticker symbols
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            lookback_days: Number of trading days if start not specified
            batch_size: Number of symbols per API call

        Returns:
            DataFrame with available data (invalid symbols are skipped)
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        end = end or datetime.now()
        start = start or (end - timedelta(days=lookback_days))

        # Clean symbols - remove any that have obvious issues
        clean_symbols = []
        for s in symbols:
            # Skip symbols with numbers in the middle (usually delisted markers)
            if any(c.isdigit() for c in s[:-1]):  # Allow digits at end like V2
                continue
            # Convert BRK.B format to Alpaca-friendly format (if needed)
            clean_symbols.append(s)

        all_records = []
        failed_symbols = []

        print(f"[DataLoader] Loading {len(clean_symbols)} symbols in batches of {batch_size}...")

        for i in range(0, len(clean_symbols), batch_size):
            batch = clean_symbols[i:i + batch_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=self.TIMEFRAME_MAP[timeframe],
                    start=start,
                    end=end,
                )
                bars = self.data_client.get_stock_bars(request)

                if bars.data:
                    for symbol, symbol_bars in bars.data.items():
                        for bar in symbol_bars:
                            all_records.append({
                                "symbol": symbol,
                                "timestamp": bar.timestamp,
                                "open": float(bar.open),
                                "high": float(bar.high),
                                "low": float(bar.low),
                                "close": float(bar.close),
                                "volume": int(bar.volume),
                                "vwap": float(bar.vwap) if bar.vwap else None,
                                "trade_count": int(bar.trade_count) if bar.trade_count else None,
                            })

                print(f"[DataLoader] Batch {i//batch_size + 1}: loaded {len(batch)} symbols")

            except Exception as e:
                # If batch fails, try loading symbols individually
                print(f"[DataLoader] Batch failed, trying individual symbols: {e}")
                for symbol in batch:
                    try:
                        request = StockBarsRequest(
                            symbol_or_symbols=[symbol],
                            timeframe=self.TIMEFRAME_MAP[timeframe],
                            start=start,
                            end=end,
                        )
                        bars = self.data_client.get_stock_bars(request)

                        if bars.data and symbol in bars.data:
                            for bar in bars.data[symbol]:
                                all_records.append({
                                    "symbol": symbol,
                                    "timestamp": bar.timestamp,
                                    "open": float(bar.open),
                                    "high": float(bar.high),
                                    "low": float(bar.low),
                                    "close": float(bar.close),
                                    "volume": int(bar.volume),
                                    "vwap": float(bar.vwap) if bar.vwap else None,
                                    "trade_count": int(bar.trade_count) if bar.trade_count else None,
                                })
                    except Exception as e2:
                        failed_symbols.append(symbol)

        if failed_symbols:
            print(f"[DataLoader] Skipped {len(failed_symbols)} invalid/unavailable symbols")

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index(["symbol", "timestamp"]).sort_index()

        print(f"[DataLoader] Total: {len(df)} bars for {df.index.get_level_values(0).nunique()} symbols")
        return df

    def prepare_backtest_data(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        lookback_days: int = 504,  # ~2 years
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data specifically formatted for backtesting.

        Args:
            symbols: List of ticker symbols
            timeframe: Bar timeframe
            lookback_days: Historical lookback period

        Returns:
            Dictionary mapping symbol to OHLCV DataFrame
        """
        bars = self.get_bars(symbols, timeframe, lookback_days=lookback_days)

        if bars.empty:
            return {}

        result = {}
        for symbol in symbols:
            if symbol in bars.index.get_level_values(0):
                symbol_data = bars.loc[symbol].copy()
                symbol_data = symbol_data.reset_index()
                symbol_data = symbol_data.rename(columns={"timestamp": "date"})
                symbol_data = symbol_data.set_index("date")
                result[symbol] = symbol_data

        return result

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return statistics.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        if df.empty:
            return {"valid": False, "error": "Empty DataFrame"}

        stats = {
            "valid": True,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": None,
            "missing_values": {},
            "warnings": [],
        }

        # Check for missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                stats["missing_values"][col] = missing
                pct = missing / len(df) * 100
                if pct > 5:
                    stats["warnings"].append(
                        f"Column '{col}' has {pct:.1f}% missing values"
                    )

        # Check date range
        if "timestamp" in df.index.names:
            dates = df.index.get_level_values("timestamp")
            stats["date_range"] = {
                "start": str(dates.min()),
                "end": str(dates.max()),
                "trading_days": len(dates.unique()),
            }

        # Check for negative prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns and (df[col] < 0).any():
                stats["warnings"].append(f"Negative values in '{col}'")
                stats["valid"] = False

        return stats


# Convenience function for quick data fetching
def load_bars(
    symbols: List[str],
    timeframe: str = "1Day",
    lookback_days: int = 252,
) -> pd.DataFrame:
    """
    Quick function to load bar data.

    Args:
        symbols: List of ticker symbols
        timeframe: Bar timeframe
        lookback_days: Historical lookback period

    Returns:
        DataFrame with OHLCV data
    """
    loader = AlpacaDataLoader()
    return loader.get_bars(symbols, timeframe, lookback_days=lookback_days)


if __name__ == "__main__":
    # Example usage
    loader = AlpacaDataLoader()

    # Get popular ETFs data
    etfs = loader.get_popular_etfs()[:10]
    print(f"Testing with ETFs: {etfs}")

    bars = loader.get_bars(etfs, timeframe="1Day", lookback_days=30)
    print(f"\nLoaded {len(bars)} bars")
    print(bars.head())

    # Validate data
    stats = loader.validate_data(bars)
    print(f"\nValidation: {stats}")
