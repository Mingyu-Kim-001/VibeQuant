"""
Dynamic Universe Module

Point-in-time universe construction:
- Select tickers based on conditions at each point in time
- Add new tickers when they IPO and meet conditions
- Remove tickers when they no longer meet conditions
- No look-ahead bias
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .alpaca_loader import AlpacaDataLoader


@dataclass
class DynamicUniverseConfig:
    """Configuration for dynamic universe."""
    min_price: float = 5.0
    max_price: float = 10000.0
    min_avg_volume: int = 500_000  # Minimum average daily volume
    min_trading_days: int = 60  # Minimum days of trading history required
    volume_lookback: int = 20  # Days to calculate average volume
    rebalance_frequency: int = 21  # How often to update universe (trading days)
    max_symbols: int = 500  # Maximum symbols in universe at any time
    exclude_leveraged: bool = True  # Exclude leveraged ETFs
    exclude_otc: bool = True  # Exclude OTC stocks


# 3x Leveraged/Inverse ETFs to exclude (keep 2x)
LEVERAGED_3X_ETFS = {
    # 3x Bull
    'TQQQ', 'SOXL', 'SPXL', 'TNA', 'NUGT', 'LABU', 'JNUG',
    'FNGU', 'ERX', 'FAS', 'TECL', 'UPRO', 'UDOW', 'TMF',
    'URTY', 'UMDD', 'YINN', 'EDC', 'CURE', 'NAIL', 'DPST',
    'DFEN', 'RETL', 'MIDU', 'WANT', 'DUSL', 'HIBL', 'WEBL',
    'BULZ', 'TPOR', 'PILL', 'UTSL',
    # 3x Bear
    'SQQQ', 'SOXS', 'SPXS', 'TZA', 'DUST', 'LABD', 'JDST',
    'FNGD', 'ERY', 'FAZ', 'TECS', 'SPXU', 'SDOW', 'TMV',
    'SRTY', 'SMDD', 'YANG', 'EDZ', 'DRIP', 'WEBS', 'HIBS',
    # Single-stock leveraged (very risky)
    'TSLL', 'TSLS', 'TSLQ', 'TSLZ', 'MSTU', 'MSTZ', 'NVDL', 'NVDS',
    'NVDU', 'NVDD', 'AMDL', 'AMDS', 'AAPU', 'AAPD', 'MSFU', 'MSFD',
    'AMZU', 'AMZD', 'GOOU', 'GOOD', 'METV', 'CONL', 'CONY',
    # Volatility products (complex)
    'UVXY', 'UVIX', 'SVXY', 'VXX', 'VIXY',
    # 3x Commodity
    'BOIL', 'KOLD', 'UNG', 'UGAZ', 'DGAZ',
}

# 2x Leveraged ETFs (allowed by default)
LEVERAGED_2X_ETFS = {
    # 2x Bull
    'SSO', 'QLD', 'UWM', 'DDM', 'MVV', 'SAA', 'UYG', 'ROM',
    'UGE', 'UCC', 'RXL', 'UXI', 'DIG', 'USD', 'UPW', 'UGL',
    'AGQ', 'UCO', 'BOIL', 'UBT', 'UST',
    # 2x Bear
    'SDS', 'QID', 'TWM', 'DXD', 'MZZ', 'SDD', 'SKF', 'REK',
    'SRS', 'SCC', 'RXD', 'SIJ', 'DUG', 'SSG', 'SDP', 'GLL',
    'ZSL', 'SCO', 'TBT', 'TBF',
    # Inverse (1x)
    'SH', 'PSQ', 'RWM', 'DOG',
}

# For backward compatibility
LEVERAGED_ETFS = LEVERAGED_3X_ETFS


class DynamicUniverseBuilder:
    """
    Builds a point-in-time universe that evolves over time.

    Key features:
    - Selects stocks based on conditions AT EACH POINT IN TIME
    - New stocks enter when they meet conditions (after min_trading_days)
    - Stocks exit when they no longer meet conditions
    - No look-ahead bias
    """

    def __init__(
        self,
        config: Optional[DynamicUniverseConfig] = None,
        cache_dir: str = "./data_cache",
    ):
        self.config = config or DynamicUniverseConfig()
        self.alpaca_loader = AlpacaDataLoader(cache_dir=cache_dir)

    def build_universe_mask(
        self,
        start_date: datetime = datetime(2016, 1, 1),
        end_date: Optional[datetime] = None,
        candidate_symbols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build a dynamic universe mask based on point-in-time conditions.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            candidate_symbols: List of candidate symbols to consider (if None, uses all liquid)

        Returns:
            Tuple of (prices DataFrame, universe_mask DataFrame)
            - prices: Close prices for all symbols
            - universe_mask: Boolean mask indicating if symbol meets conditions on each date
        """
        end_date = end_date or datetime.now()

        # Get candidate symbols (load broadly to have historical data)
        if candidate_symbols is None:
            print("[DynamicUniverse] Getting candidate symbols...")
            candidate_symbols = self.alpaca_loader.get_liquid_universe(
                min_volume=100_000,  # Lower threshold to get more candidates
                max_symbols=2000,
            )

        # Filter out leveraged ETFs if configured
        if self.config.exclude_leveraged:
            original_count = len(candidate_symbols)
            candidate_symbols = [s for s in candidate_symbols if s not in LEVERAGED_ETFS]
            print(f"[DynamicUniverse] Excluded {original_count - len(candidate_symbols)} leveraged ETFs")

        print(f"[DynamicUniverse] Loading data for {len(candidate_symbols)} candidate symbols...")

        # Load all data
        data = self.alpaca_loader.get_bars_safe(
            symbols=candidate_symbols,
            timeframe="1Day",
            start=start_date - timedelta(days=90),  # Extra days for lookback
            end=end_date,
            batch_size=100,
        )

        if data.empty:
            raise ValueError("No data loaded")

        # Pivot to wide format
        prices = data["close"].unstack(level=0)
        volumes = data["volume"].unstack(level=0)

        # Filter to requested date range
        prices = prices[prices.index >= pd.Timestamp(start_date).tz_localize('UTC')]
        volumes = volumes[volumes.index >= pd.Timestamp(start_date).tz_localize('UTC')]

        print(f"[DynamicUniverse] Price data shape: {prices.shape}")
        print(f"[DynamicUniverse] Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

        # Build dynamic universe mask
        print("[DynamicUniverse] Building point-in-time universe mask...")
        universe_mask = self._build_mask(prices, volumes)

        # Count stats
        valid_counts = universe_mask.sum(axis=1)
        print(f"[DynamicUniverse] Universe size over time: min={valid_counts.min()}, "
              f"max={valid_counts.max()}, avg={valid_counts.mean():.0f}")

        return prices, universe_mask

    def _build_mask(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the dynamic universe mask based on point-in-time conditions.

        For each date, a stock is in the universe if:
        1. It has at least min_trading_days of history
        2. Price is between min_price and max_price
        3. Average volume over lookback period >= min_avg_volume
        4. The stock has valid (non-NaN) data
        """
        universe_mask = pd.DataFrame(
            False,
            index=prices.index,
            columns=prices.columns,
        )

        # Calculate rolling average volume
        avg_volume = volumes.rolling(
            window=self.config.volume_lookback,
            min_periods=self.config.volume_lookback,
        ).mean()

        # Calculate trading days since first data
        first_valid_date = {}
        for col in prices.columns:
            valid_dates = prices[col].dropna().index
            if len(valid_dates) > 0:
                first_valid_date[col] = valid_dates[0]

        # Build mask for each rebalance date
        rebalance_dates = prices.index[::self.config.rebalance_frequency]

        current_universe = set()

        for i, rebal_date in enumerate(rebalance_dates):
            # Determine which stocks meet conditions at this date
            qualifying_stocks = []

            for col in prices.columns:
                # Skip if no data yet
                if col not in first_valid_date:
                    continue

                # Check minimum trading days
                days_since_first = (rebal_date - first_valid_date[col]).days
                if days_since_first < self.config.min_trading_days:
                    continue

                # Check if we have valid data at this date
                if pd.isna(prices.loc[rebal_date, col]):
                    continue

                price = prices.loc[rebal_date, col]
                vol = avg_volume.loc[rebal_date, col] if rebal_date in avg_volume.index else np.nan

                # Check price bounds
                if not (self.config.min_price <= price <= self.config.max_price):
                    continue

                # Check volume threshold
                if pd.isna(vol) or vol < self.config.min_avg_volume:
                    continue

                qualifying_stocks.append((col, vol))

            # Sort by volume and take top max_symbols
            qualifying_stocks.sort(key=lambda x: x[1], reverse=True)
            new_universe = set(s[0] for s in qualifying_stocks[:self.config.max_symbols])

            # Fill mask from this rebalance date to the next
            if i < len(rebalance_dates) - 1:
                next_rebal = rebalance_dates[i + 1]
                mask_slice = (prices.index >= rebal_date) & (prices.index < next_rebal)
            else:
                mask_slice = prices.index >= rebal_date

            for sym in new_universe:
                universe_mask.loc[mask_slice, sym] = True

            # Log changes
            added = new_universe - current_universe
            removed = current_universe - new_universe
            if added or removed:
                if len(added) > 0:
                    print(f"[{rebal_date.date()}] +{len(added)} stocks entered universe")
                if len(removed) > 0:
                    print(f"[{rebal_date.date()}] -{len(removed)} stocks exited universe")

            current_universe = new_universe

        return universe_mask

    def get_universe_at_date(
        self,
        date: datetime,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> List[str]:
        """
        Get the universe of stocks that meet conditions at a specific date.

        This is useful for real-time trading to determine current universe.
        """
        date_ts = pd.Timestamp(date)
        if date_ts.tz is None:
            date_ts = date_ts.tz_localize('UTC')

        qualifying = []

        # Calculate average volume up to this date
        vol_lookback = volumes.loc[:date_ts].tail(self.config.volume_lookback)
        avg_vol = vol_lookback.mean()

        for col in prices.columns:
            # Check if we have data
            if pd.isna(prices.loc[date_ts, col]):
                continue

            # Check first valid date
            first_valid = prices[col].dropna().index[0]
            days_trading = (date_ts - first_valid).days
            if days_trading < self.config.min_trading_days:
                continue

            price = prices.loc[date_ts, col]
            vol = avg_vol[col] if col in avg_vol else 0

            # Check conditions
            if (self.config.min_price <= price <= self.config.max_price and
                vol >= self.config.min_avg_volume):
                qualifying.append((col, vol))

        # Sort by volume, return top symbols
        qualifying.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in qualifying[:self.config.max_symbols]]


def build_dynamic_universe(
    start_date: datetime = datetime(2016, 1, 1),
    end_date: Optional[datetime] = None,
    min_volume: int = 500_000,
    max_symbols: int = 500,
    exclude_leveraged: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to build a dynamic, point-in-time universe.

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        min_volume: Minimum average daily volume
        max_symbols: Maximum symbols in universe
        exclude_leveraged: Whether to exclude leveraged ETFs

    Returns:
        Tuple of (prices, universe_mask)
    """
    config = DynamicUniverseConfig(
        min_avg_volume=min_volume,
        max_symbols=max_symbols,
        exclude_leveraged=exclude_leveraged,
    )
    builder = DynamicUniverseBuilder(config)
    return builder.build_universe_mask(start_date, end_date)


if __name__ == "__main__":
    # Test dynamic universe
    print("Building dynamic universe...")
    prices, mask = build_dynamic_universe(
        start_date=datetime(2016, 1, 1),
        min_volume=500_000,
        max_symbols=200,
        exclude_leveraged=True,
    )

    print(f"\nFinal prices shape: {prices.shape}")
    print(f"Universe mask shape: {mask.shape}")

    # Show universe evolution
    monthly_counts = mask.resample('M').mean().mean(axis=1) * mask.shape[1]
    print("\nAverage universe size by month (first 12 months):")
    print(monthly_counts.head(12))
