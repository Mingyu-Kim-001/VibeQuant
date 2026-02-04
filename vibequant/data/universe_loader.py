"""
Universe Loader Module

Supports multiple ticker universes:
- S&P 500 (with survivorship-free option)
- Russell 1000/2000/3000
- NASDAQ 100
- All liquid US stocks
- Custom CSV files
- Exchange-specific (NYSE, NASDAQ)
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .alpaca_loader import AlpacaDataLoader


@dataclass
class UniverseInfo:
    """Information about a universe."""
    name: str
    description: str
    num_symbols: int
    symbols: List[str]
    survivorship_free: bool = False
    source: str = "alpaca"


class UniverseLoader:
    """
    Flexible universe loader supporting multiple sources.
    """

    # Well-known index constituents (approximate/static lists)
    NASDAQ_100 = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "AVGO", "COST",
        "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC", "INTU", "CMCSA", "TXN",
        "AMGN", "QCOM", "HON", "AMAT", "ISRG", "BKNG", "SBUX", "VRTX", "MDLZ", "ADP",
        "GILD", "LRCX", "ADI", "REGN", "PANW", "MU", "KLAC", "SNPS", "CDNS", "PYPL",
        "MELI", "MAR", "ASML", "ORLY", "CTAS", "MNST", "CSX", "FTNT", "NXPI", "PCAR",
        "WDAY", "DXCM", "MRVL", "CPRT", "ROST", "KDP", "AEP", "PAYX", "ODFL", "KHC",
        "CHTR", "MRNA", "IDXX", "FAST", "EA", "EXC", "GEHC", "LULU", "VRSK", "CTSH",
        "XEL", "CSGP", "BKR", "ANSS", "DDOG", "FANG", "ZS", "TTWO", "DLTR", "TEAM",
        "WBD", "ILMN", "BIIB", "ALGN", "WBA", "SIRI", "ENPH", "CEG", "ON", "ARM",
        "CRWD", "MDB", "DASH", "CDW", "GFS", "SMCI", "MCHP", "SPLK", "CCEP", "PDD",
    ]

    RUSSELL_1000_SAMPLE = [
        # Top ~200 from Russell 1000 (approximation)
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK.B", "UNH", "XOM", "JNJ",
        "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "AVGO",
        "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT", "DHR", "NEE", "VZ", "ADBE",
        "TXN", "PM", "CMCSA", "NKE", "RTX", "NFLX", "BMY", "HON", "UNP", "ORCL", "QCOM",
        "COP", "LOW", "UPS", "MS", "SPGI", "IBM", "INTC", "CAT", "BA", "GE", "SBUX",
        "DE", "INTU", "AMD", "AMGN", "GS", "BLK", "MDLZ", "ISRG", "ADP", "GILD", "ADI",
        "SYK", "C", "REGN", "T", "LMT", "PLD", "BKNG", "VRTX", "TMUS", "ZTS", "CI",
        "SCHW", "CVS", "CB", "SO", "MMC", "DUK", "MO", "PNC", "TJX", "CL", "BDX",
        "ITW", "USB", "TGT", "EOG", "SLB", "PSA", "CME", "AON", "NOC", "ICE", "WM",
        "FCX", "NSC", "FDX", "EMR", "GD", "APD", "MCK", "CCI", "D", "EW", "SRE",
        # Add more mid-caps
        "CARR", "PWR", "CBRE", "FICO", "WST", "AXON", "LYV", "EQR", "VTR", "DECK",
        "WAB", "IR", "VICI", "EXR", "RCL", "GWW", "DRI", "STE", "TER", "DOV",
        "ULTA", "ZBRA", "WAT", "BR", "BAX", "PKG", "EXPD", "DGX", "MKC", "TDY",
        "SWKS", "POOL", "SNA", "NTAP", "CF", "NVR", "AOS", "JBHT", "RJF", "TECH",
        "HWM", "TPR", "GNRC", "EPAM", "MPWR", "MANH", "PODD", "PAYC", "TYL", "ALLE",
    ]

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.alpaca_loader = AlpacaDataLoader(cache_dir=cache_dir)

    def get_available_universes(self) -> Dict[str, str]:
        """Get list of available universe types."""
        return {
            "sp500": "S&P 500 constituents (with survivorship-free option)",
            "nasdaq100": "NASDAQ 100 constituents",
            "russell1000": "Russell 1000 (large-cap, ~1000 stocks)",
            "liquid_500": "Top 500 most liquid US stocks",
            "liquid_1000": "Top 1000 most liquid US stocks",
            "liquid_2000": "Top 2000 most liquid US stocks",
            "all_liquid": "All liquid stocks (volume > 100K)",
            "nyse": "NYSE-listed stocks only",
            "nasdaq": "NASDAQ-listed stocks only",
            "etfs": "ETFs only",
            "custom": "Custom CSV file",
        }

    def load_universe(
        self,
        universe_type: str,
        max_symbols: int = 0,
        min_volume: int = 500_000,
        custom_file: Optional[str] = None,
        survivorship_free: bool = False,
    ) -> UniverseInfo:
        """
        Load a universe of symbols.

        Args:
            universe_type: Type of universe (see get_available_universes())
            max_symbols: Maximum number of symbols (0 = no limit)
            min_volume: Minimum average daily volume filter
            custom_file: Path to custom CSV file (for 'custom' type)
            survivorship_free: Whether to use survivorship-free data (S&P 500 only)

        Returns:
            UniverseInfo with symbols and metadata
        """
        if universe_type == "sp500":
            return self._load_sp500(max_symbols, survivorship_free)
        elif universe_type == "nasdaq100":
            return self._load_nasdaq100()
        elif universe_type == "russell1000":
            return self._load_russell1000(max_symbols)
        elif universe_type.startswith("liquid_"):
            n = int(universe_type.split("_")[1])
            return self._load_liquid(n, min_volume)
        elif universe_type == "all_liquid":
            return self._load_liquid(5000, min_volume)
        elif universe_type == "nyse":
            return self._load_by_exchange("NYSE", max_symbols, min_volume)
        elif universe_type == "nasdaq":
            return self._load_by_exchange("NASDAQ", max_symbols, min_volume)
        elif universe_type == "etfs":
            return self._load_etfs()
        elif universe_type == "custom":
            if not custom_file:
                raise ValueError("custom_file required for 'custom' universe type")
            return self._load_custom(custom_file)
        else:
            raise ValueError(f"Unknown universe type: {universe_type}")

    def _load_sp500(self, max_symbols: int, survivorship_free: bool) -> UniverseInfo:
        """Load S&P 500 universe."""
        if survivorship_free:
            from .sp500_loader import SP500SurvivorshipFreeLoader
            loader = SP500SurvivorshipFreeLoader(
                membership_csv_path=str(self.cache_dir / "sp500_membership.csv"),
                cache_dir=str(self.cache_dir),
            )
            symbols = loader.get_active_tickers()
            if max_symbols > 0:
                symbols = symbols[:max_symbols]
            return UniverseInfo(
                name="S&P 500 (Survivorship-Free)",
                description="S&P 500 with actual membership dates from github.com/fja05680/sp500",
                num_symbols=len(symbols),
                symbols=symbols,
                survivorship_free=True,
                source="sp500_membership.csv",
            )
        else:
            symbols = self.alpaca_loader.get_sp500_symbols()
            if max_symbols > 0:
                symbols = symbols[:max_symbols]
            return UniverseInfo(
                name="S&P 500",
                description="Current S&P 500 constituents (contains survivorship bias)",
                num_symbols=len(symbols),
                symbols=symbols,
                source="alpaca",
            )

    def _load_nasdaq100(self) -> UniverseInfo:
        """Load NASDAQ 100 universe."""
        return UniverseInfo(
            name="NASDAQ 100",
            description="NASDAQ 100 index constituents",
            num_symbols=len(self.NASDAQ_100),
            symbols=self.NASDAQ_100,
            source="static",
        )

    def _load_russell1000(self, max_symbols: int) -> UniverseInfo:
        """Load Russell 1000 universe (approximation)."""
        symbols = self.RUSSELL_1000_SAMPLE
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        return UniverseInfo(
            name="Russell 1000 (Sample)",
            description="Russell 1000 large-cap stocks (representative sample)",
            num_symbols=len(symbols),
            symbols=symbols,
            source="static",
        )

    def _load_liquid(self, n: int, min_volume: int) -> UniverseInfo:
        """Load top N liquid stocks."""
        symbols = self.alpaca_loader.get_liquid_universe(
            min_volume=min_volume,
            max_symbols=n,
        )
        return UniverseInfo(
            name=f"Liquid {n}",
            description=f"Top {n} most liquid stocks (vol > {min_volume:,})",
            num_symbols=len(symbols),
            symbols=symbols,
            source="alpaca",
        )

    def _load_by_exchange(self, exchange: str, max_symbols: int, min_volume: int) -> UniverseInfo:
        """Load stocks from a specific exchange."""
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus

        trading_client = TradingClient(
            self.alpaca_loader.api_key,
            self.alpaca_loader.secret_key,
        )

        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = trading_client.get_all_assets(request)

        # Filter by exchange
        exchange_assets = [
            a.symbol for a in assets
            if a.tradable and str(a.exchange) == exchange
        ]

        # Get liquid subset
        if len(exchange_assets) > 200:
            # Need to filter by volume
            liquid = self.alpaca_loader.get_liquid_universe(
                min_volume=min_volume,
                max_symbols=5000,
            )
            symbols = [s for s in liquid if s in exchange_assets]
        else:
            symbols = exchange_assets

        if max_symbols > 0:
            symbols = symbols[:max_symbols]

        return UniverseInfo(
            name=f"{exchange} Stocks",
            description=f"Stocks listed on {exchange}",
            num_symbols=len(symbols),
            symbols=symbols,
            source="alpaca",
        )

    def _load_etfs(self) -> UniverseInfo:
        """Load ETFs only."""
        etfs = self.alpaca_loader.get_popular_etfs()
        return UniverseInfo(
            name="ETFs",
            description="Popular ETFs (sector, broad market, commodities)",
            num_symbols=len(etfs),
            symbols=etfs,
            source="static",
        )

    def _load_custom(self, filepath: str) -> UniverseInfo:
        """Load universe from custom CSV file."""
        df = pd.read_csv(filepath)

        # Try to find symbol column
        symbol_col = None
        for col in ['Symbol', 'symbol', 'SYMBOL', 'ticker', 'Ticker', 'TICKER']:
            if col in df.columns:
                symbol_col = col
                break

        if symbol_col is None:
            # Assume first column is symbols
            symbol_col = df.columns[0]

        symbols = df[symbol_col].dropna().astype(str).str.strip().tolist()
        # Clean symbols - remove any with special characters
        symbols = [s for s in symbols if s.isalpha() or '.' in s]

        return UniverseInfo(
            name=f"Custom ({Path(filepath).stem})",
            description=f"Custom universe from {filepath}",
            num_symbols=len(symbols),
            symbols=symbols,
            source=filepath,
        )

    def combine_universes(
        self,
        universe_types: List[str],
        deduplicate: bool = True,
    ) -> UniverseInfo:
        """
        Combine multiple universes into one.

        Args:
            universe_types: List of universe types to combine
            deduplicate: Remove duplicate symbols

        Returns:
            Combined UniverseInfo
        """
        all_symbols = []
        names = []

        for ut in universe_types:
            info = self.load_universe(ut)
            all_symbols.extend(info.symbols)
            names.append(info.name)

        if deduplicate:
            all_symbols = list(dict.fromkeys(all_symbols))  # Preserve order

        return UniverseInfo(
            name=" + ".join(names),
            description=f"Combined universe: {', '.join(names)}",
            num_symbols=len(all_symbols),
            symbols=all_symbols,
            source="combined",
        )


def load_universe_with_data(
    universe_type: str,
    start_date: datetime = datetime(2016, 1, 1),
    end_date: Optional[datetime] = None,
    max_symbols: int = 200,
    cache_dir: str = "./data_cache",
) -> Tuple[pd.DataFrame, pd.DataFrame, UniverseInfo]:
    """
    Load any universe with price data and universe mask.

    This is the RECOMMENDED way to load data for backtesting.
    It supports multiple universes with clear bias labeling.

    Args:
        universe_type: One of:
            - "sp500_sf" : S&P 500 survivorship-free (RECOMMENDED for realistic backtests)
            - "sp500"    : S&P 500 current constituents (biased)
            - "nasdaq100": NASDAQ 100 (biased)
            - "liquid_N" : Top N liquid stocks (biased, e.g., "liquid_200")
            - "dynamic"  : Point-in-time universe (least biased for non-index)
            - "etfs"     : ETFs only
        start_date: Backtest start date
        end_date: Backtest end date (default: now)
        max_symbols: Maximum symbols to include
        cache_dir: Directory for caching data

    Returns:
        Tuple of (prices, universe_mask, universe_info)
        - prices: DataFrame of close prices
        - universe_mask: Boolean DataFrame indicating valid symbols per date
        - universe_info: Metadata about the universe

    Example:
        >>> prices, mask, info = load_universe_with_data("sp500_sf", max_symbols=200)
        >>> print(f"Loaded {info.name}: {info.num_symbols} symbols")
        >>> print(f"Survivorship-free: {info.survivorship_free}")
    """
    end_date = end_date or datetime.now()
    loader = UniverseLoader(cache_dir=cache_dir)

    # Handle different universe types
    if universe_type == "sp500_sf":
        # Survivorship-free S&P 500 (RECOMMENDED)
        from .sp500_loader import SP500SurvivorshipFreeLoader
        sp500_loader = SP500SurvivorshipFreeLoader(
            membership_csv_path=f"{cache_dir}/sp500_membership.csv",
            cache_dir=cache_dir,
        )
        prices, mask = sp500_loader.load_survivorship_free_data(
            start_date=start_date,
            end_date=end_date,
            include_etfs=True,
            max_symbols=max_symbols,
        )
        info = UniverseInfo(
            name="S&P 500 Survivorship-Free",
            description="S&P 500 with actual membership dates - most realistic",
            num_symbols=len(prices.columns),
            symbols=prices.columns.tolist(),
            survivorship_free=True,
            source="sp500_membership.csv",
        )
        return prices, mask, info

    elif universe_type == "dynamic":
        # Dynamic point-in-time universe
        from .dynamic_universe import build_dynamic_universe
        prices, mask = build_dynamic_universe(
            start_date=start_date,
            end_date=end_date,
            min_volume=500_000,
            max_symbols=max_symbols,
            exclude_leveraged=True,
        )
        info = UniverseInfo(
            name="Dynamic Universe",
            description="Point-in-time selection, no look-ahead bias",
            num_symbols=len(prices.columns),
            symbols=prices.columns.tolist(),
            survivorship_free=True,  # Dynamic is bias-free
            source="dynamic",
        )
        return prices, mask, info

    else:
        # Other universes (may have bias)
        universe_info = loader.load_universe(
            universe_type=universe_type,
            max_symbols=max_symbols,
            survivorship_free=False,
        )

        # Load price data
        alpaca = AlpacaDataLoader(cache_dir=cache_dir)
        data = alpaca.get_bars_safe(
            symbols=universe_info.symbols,
            timeframe="1Day",
            start=start_date,
            end=end_date,
            batch_size=100,
        )

        if data.empty:
            raise ValueError(f"No data loaded for {universe_type}")

        prices = data["close"].unstack(level=0)

        # Create simple mask (all True where data exists)
        mask = ~prices.isna()

        # Update info with actual symbols loaded
        universe_info.num_symbols = len(prices.columns)
        universe_info.symbols = prices.columns.tolist()

        # Add bias warning to description
        if not universe_info.survivorship_free:
            universe_info.description += " [WARNING: Contains survivorship bias]"

        return prices, mask, universe_info


# Mapping of universe keys to descriptions for easy reference
UNIVERSE_TYPES = {
    "sp500_sf": {
        "name": "S&P 500 Survivorship-Free",
        "biased": False,
        "description": "Uses actual S&P 500 membership dates. TSLA valid from Dec 21, 2020, not 2010.",
        "recommended": True,
    },
    "sp500": {
        "name": "S&P 500 (Current)",
        "biased": True,
        "description": "Current S&P 500 constituents applied historically. Contains survivorship bias.",
        "recommended": False,
    },
    "dynamic": {
        "name": "Dynamic Universe",
        "biased": False,
        "description": "Point-in-time selection based on price/volume criteria. No look-ahead bias.",
        "recommended": True,
    },
    "nasdaq100": {
        "name": "NASDAQ 100",
        "biased": True,
        "description": "Current NASDAQ 100 constituents. Contains survivorship bias.",
        "recommended": False,
    },
    "liquid_200": {
        "name": "Liquid 200",
        "biased": True,
        "description": "Top 200 most liquid stocks today. High survivorship bias.",
        "recommended": False,
    },
    "liquid_500": {
        "name": "Liquid 500",
        "biased": True,
        "description": "Top 500 most liquid stocks today. High survivorship bias.",
        "recommended": False,
    },
    "etfs": {
        "name": "ETFs Only",
        "biased": False,
        "description": "ETFs only. No single-stock risk, minimal survivorship bias.",
        "recommended": True,
    },
}


def print_universe_summary():
    """Print summary of available universes."""
    print("=" * 70)
    print("AVAILABLE UNIVERSES FOR BACKTESTING")
    print("=" * 70)

    print("\n** RECOMMENDED (No/Low Bias) **")
    for key, info in UNIVERSE_TYPES.items():
        if info["recommended"]:
            bias_label = "No Bias" if not info["biased"] else "Low Bias"
            print(f"  {key:15} - {info['name']} [{bias_label}]")
            print(f"                    {info['description']}")

    print("\n** BIASED (Use for exploration only) **")
    for key, info in UNIVERSE_TYPES.items():
        if not info["recommended"]:
            print(f"  {key:15} - {info['name']} [BIASED]")
            print(f"                    {info['description']}")

    print("\n" + "=" * 70)
    print("USAGE:")
    print("  from vibequant.data import load_universe_with_data")
    print("  prices, mask, info = load_universe_with_data('sp500_sf', max_symbols=200)")
    print("=" * 70)


if __name__ == "__main__":
    print_universe_summary()
