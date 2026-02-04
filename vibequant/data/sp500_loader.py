"""
S&P 500 Survivorship-Bias-Free Data Loader

Loads S&P 500 constituents with proper handling of additions/removals
to avoid survivorship bias in backtesting.

Uses actual S&P 500 membership dates from:
https://github.com/fja05680/sp500
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .alpaca_loader import AlpacaDataLoader


class SP500SurvivorshipFreeLoader:
    """
    Loads S&P 500 data without survivorship bias.

    Uses historical constituent data to only include stocks that were
    actually in the S&P 500 at each point in time.

    Data source: https://github.com/fja05680/sp500
    """

    def __init__(
        self,
        membership_csv_path: str = "./data_cache/sp500_membership.csv",
        cache_dir: str = "./data_cache",
    ):
        """
        Initialize the S&P 500 survivorship-free loader.

        Args:
            membership_csv_path: Path to sp500_membership.csv with actual S&P 500
                                 addition/removal dates. Download from:
                                 https://github.com/fja05680/sp500
            cache_dir: Directory for caching price data from Alpaca
        """
        self.membership_csv_path = Path(membership_csv_path)
        self.cache_dir = Path(cache_dir)
        self.alpaca_loader = AlpacaDataLoader(cache_dir=cache_dir)

        # Load S&P 500 membership data (actual index inclusion dates)
        self.membership = self._load_membership()

    def _load_membership(self) -> pd.DataFrame:
        """
        Load S&P 500 membership dates.

        This file contains the actual dates when stocks were added to
        and removed from the S&P 500 index.

        Source: https://github.com/fja05680/sp500/blob/master/sp500_ticker_start_end.csv
        """
        if not self.membership_csv_path.exists():
            print(f"[SP500Loader] WARNING: Membership file not found: {self.membership_csv_path}")
            print("[SP500Loader] Using price data dates as fallback (less accurate)")
            return pd.DataFrame()

        df = pd.read_csv(self.membership_csv_path)

        # Parse dates
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        # For stocks still in index, end_date is NaN - set to far future
        df['end_date'] = df['end_date'].fillna(datetime(2099, 12, 31))

        # Mark if currently active
        df['is_active'] = df['end_date'] >= datetime(2025, 1, 1)

        print(f"[SP500Loader] Loaded {len(df)} S&P 500 membership records")
        print(f"[SP500Loader] Unique tickers: {df['ticker'].nunique()}")
        print(f"[SP500Loader] Currently active: {df['is_active'].sum()}")
        print(f"[SP500Loader] Date range: {df['start_date'].min().date()} to {df['end_date'].max().date()}")

        # Show some examples
        examples = ['TSLA', 'META', 'NVDA', 'AAPL']
        print(f"[SP500Loader] Example membership dates:")
        for ticker in examples:
            rows = df[df['ticker'] == ticker]
            if not rows.empty:
                for _, row in rows.iterrows():
                    end = 'Active' if row['is_active'] else row['end_date'].strftime('%Y-%m-%d')
                    print(f"  {ticker}: {row['start_date'].strftime('%Y-%m-%d')} → {end}")

        return df

    def get_all_tickers(self) -> List[str]:
        """Get all unique tickers from membership data."""
        if self.membership.empty:
            raise ValueError(
                "Membership data not loaded. Download sp500_membership.csv from "
                "https://github.com/fja05680/sp500 and place in data_cache/"
            )
        return self.membership['ticker'].unique().tolist()

    def get_active_tickers(self) -> List[str]:
        """Get currently active S&P 500 constituents."""
        if self.membership.empty:
            raise ValueError(
                "Membership data not loaded. Download sp500_membership.csv from "
                "https://github.com/fja05680/sp500 and place in data_cache/"
            )
        return self.membership[self.membership['is_active']]['ticker'].unique().tolist()

    def get_tickers_at_date(self, date: datetime) -> List[str]:
        """Get list of S&P 500 constituents that were in the index on a given date."""
        if self.membership.empty:
            raise ValueError(
                "Membership data not loaded. Download sp500_membership.csv from "
                "https://github.com/fja05680/sp500 and place in data_cache/"
            )

        date_ts = pd.Timestamp(date)
        mask = (
            (self.membership['start_date'] <= date_ts) &
            (self.membership['end_date'] >= date_ts)
        )
        return self.membership[mask]['ticker'].unique().tolist()

    def load_survivorship_free_data(
        self,
        start_date: datetime = datetime(2016, 1, 1),
        end_date: Optional[datetime] = None,
        include_etfs: bool = True,
        max_symbols: int = 0,
        min_avg_volume: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load price data with survivorship-bias-free universe mask.

        The mask uses ACTUAL S&P 500 membership dates, not price availability.
        For example, TSLA is only marked as valid from Dec 21, 2020 (when it
        was actually added to the S&P 500), not from its 2010 IPO.

        Args:
            start_date: Start date for data
            end_date: End date for data
            include_etfs: Whether to include ETFs
            max_symbols: If > 0, filter to top N most liquid symbols
            min_avg_volume: If > 0, filter to symbols with average volume above threshold

        Returns:
            Tuple of (prices DataFrame, universe_mask DataFrame)
            - prices: Close prices for all symbols
            - universe_mask: Boolean mask indicating if symbol was in S&P 500 on each date
        """
        end_date = end_date or datetime.now()

        # Get all tickers from membership data
        all_tickers = self.get_all_tickers()

        # Add ETFs if requested
        if include_etfs:
            etfs = self.alpaca_loader.get_popular_etfs()
            all_tickers = list(set(all_tickers + etfs))

        print(f"[SP500Loader] Loading data for {len(all_tickers)} symbols...")

        # Load data from Alpaca
        data = self.alpaca_loader.get_bars_safe(
            symbols=all_tickers,
            timeframe="1Day",
            start=start_date,
            end=end_date,
            batch_size=100,
        )

        if data.empty:
            raise ValueError("No data loaded from Alpaca")

        # Pivot to wide format
        prices = data["close"].unstack(level=0)
        volumes = data["volume"].unstack(level=0)

        print(f"[SP500Loader] Raw price data shape: {prices.shape}")
        print(f"[SP500Loader] Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

        # Apply liquidity filtering if requested
        if max_symbols > 0 or min_avg_volume > 0:
            print("[SP500Loader] Applying liquidity filter...")

            # Calculate average daily volume
            avg_volumes = volumes.mean().sort_values(ascending=False)

            # Get ETF symbols
            etf_symbols = set(self.alpaca_loader.get_popular_etfs()) if include_etfs else set()

            # Filter by minimum volume
            if min_avg_volume > 0:
                liquid_symbols = avg_volumes[avg_volumes >= min_avg_volume].index.tolist()
            else:
                liquid_symbols = avg_volumes.index.tolist()

            # Filter to top N most liquid (preserving ETFs)
            if max_symbols > 0:
                stock_symbols = [s for s in liquid_symbols if s not in etf_symbols]
                etf_in_data = [s for s in liquid_symbols if s in etf_symbols]

                num_stock_slots = max(0, max_symbols - len(etf_in_data))
                top_stocks = stock_symbols[:num_stock_slots]

                liquid_symbols = top_stocks + etf_in_data

            # Filter prices and volumes
            common_symbols = [s for s in liquid_symbols if s in prices.columns]
            prices = prices[common_symbols]
            volumes = volumes[common_symbols]

            print(f"[SP500Loader] After liquidity filter: {len(common_symbols)} symbols")
            print(f"[SP500Loader]   - Stocks: {len([s for s in common_symbols if s not in etf_symbols])}")
            print(f"[SP500Loader]   - ETFs: {len([s for s in common_symbols if s in etf_symbols])}")

        print(f"[SP500Loader] Final price data shape: {prices.shape}")

        # Build universe mask using ACTUAL membership dates
        print("[SP500Loader] Building survivorship-free universe mask (using actual membership dates)...")
        universe_mask = pd.DataFrame(
            False,
            index=prices.index,
            columns=prices.columns
        )

        if not self.membership.empty:
            # Use actual S&P 500 membership dates
            for _, row in self.membership.iterrows():
                ticker = row['ticker']
                if ticker in universe_mask.columns:
                    # Convert to timezone-aware
                    start = pd.Timestamp(row['start_date'])
                    end = pd.Timestamp(row['end_date'])

                    if prices.index.tz is not None:
                        start = start.tz_localize(prices.index.tz)
                        end = end.tz_localize(prices.index.tz)

                    mask = (universe_mask.index >= start) & (universe_mask.index <= end)
                    universe_mask.loc[mask, ticker] = True
        else:
            # Fallback: use price availability (less accurate)
            print("[SP500Loader] WARNING: Using price availability dates (membership data not available)")
            for col in prices.columns:
                # Mark as valid where we have price data
                universe_mask[col] = ~prices[col].isna()

        # ETFs are always valid
        if include_etfs:
            etfs = self.alpaca_loader.get_popular_etfs()
            for etf in etfs:
                if etf in universe_mask.columns:
                    universe_mask[etf] = True

        # Count valid symbols per day
        valid_counts = universe_mask.sum(axis=1)
        print(f"[SP500Loader] Valid symbols per day: min={valid_counts.min()}, max={valid_counts.max()}, avg={valid_counts.mean():.0f}")

        # Show membership check for key stocks
        self._verify_membership(universe_mask)

        return prices, universe_mask

    def _verify_membership(self, universe_mask: pd.DataFrame):
        """Verify membership dates for key stocks."""
        checks = {
            'TSLA': '2020-12-21',  # Added Dec 21, 2020
            'META': '2022-06-09',  # Added Jun 9, 2022 (after FB→META rename)
            'NVDA': '2001-11-30',  # Added Nov 30, 2001
        }

        print("[SP500Loader] Membership verification:")
        for ticker, expected_start in checks.items():
            if ticker not in universe_mask.columns:
                continue

            # Find first True date
            valid_dates = universe_mask.index[universe_mask[ticker]]
            if len(valid_dates) > 0:
                first_valid = valid_dates[0].strftime('%Y-%m-%d')
                status = "✓" if first_valid >= expected_start else "✗"
                print(f"  {ticker}: First valid {first_valid} (expected >= {expected_start}) {status}")
            else:
                print(f"  {ticker}: No valid dates found")

    def apply_universe_mask(
        self,
        signals: pd.DataFrame,
        universe_mask: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply universe mask to signals - zero out signals for stocks not in universe.

        Args:
            signals: Strategy signals DataFrame
            universe_mask: Boolean mask of valid universe

        Returns:
            Masked signals DataFrame
        """
        # Align columns
        common_cols = signals.columns.intersection(universe_mask.columns)
        signals_aligned = signals[common_cols]
        mask_aligned = universe_mask[common_cols]

        # Zero out signals for stocks not in universe
        masked_signals = signals_aligned.where(mask_aligned, 0)

        # Re-normalize weights
        row_sums = masked_signals.sum(axis=1).replace(0, 1)
        masked_signals = masked_signals.div(row_sums, axis=0)

        return masked_signals.fillna(0)


def load_sp500_survivorship_free(
    start_date: datetime = datetime(2016, 1, 1),
    include_etfs: bool = True,
    max_symbols: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, SP500SurvivorshipFreeLoader]:
    """
    Convenience function to load S&P 500 data without survivorship bias.

    Returns:
        Tuple of (prices, universe_mask, loader)
    """
    loader = SP500SurvivorshipFreeLoader()
    prices, universe_mask = loader.load_survivorship_free_data(
        start_date=start_date,
        include_etfs=include_etfs,
        max_symbols=max_symbols,
    )
    return prices, universe_mask, loader


def download_membership_data(output_path: str = "./data_cache/sp500_membership.csv"):
    """
    Download latest S&P 500 membership data from GitHub.

    Source: https://github.com/fja05680/sp500
    """
    import urllib.request

    url = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
    print(f"Downloading S&P 500 membership data from {url}...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)

    print(f"Saved to {output_path}")

    # Verify
    df = pd.read_csv(output_path)
    print(f"Downloaded {len(df)} records")


if __name__ == "__main__":
    # Test the loader
    print("Testing S&P 500 Survivorship-Free Loader")
    print("=" * 60)

    loader = SP500SurvivorshipFreeLoader()

    # Check membership at different dates
    print("\nUniverse size at different dates:")
    for date in [datetime(2016, 1, 1), datetime(2020, 1, 1), datetime(2020, 12, 20), datetime(2020, 12, 22), datetime(2024, 1, 1)]:
        universe = loader.get_tickers_at_date(date)
        tsla_in = "TSLA" in universe
        meta_in = "META" in universe
        print(f"  {date.date()}: {len(universe)} stocks (TSLA: {tsla_in}, META: {meta_in})")

    # Load data with mask
    print("\nLoading data...")
    prices, mask = loader.load_survivorship_free_data(
        start_date=datetime(2016, 1, 1),
        include_etfs=True,
        max_symbols=200,
    )
    print(f"\nFinal data shape: {prices.shape}")
