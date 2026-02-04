"""
Universe Search Module

Automatically tests strategies across multiple universes to find optimal fit.
Common quant universes:
- Russell 1000 (large-cap)
- Russell 2000 (small-cap)
- Russell 3000 (broad market)
- S&P 500
- S&P 400 (mid-cap)
- S&P 600 (small-cap)
- NASDAQ 100
- All liquid stocks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass
import json
import os

from .data import AlpacaDataLoader
from .data.dynamic_universe import DynamicUniverseBuilder, DynamicUniverseConfig, LEVERAGED_3X_ETFS
from .backtest import BacktestEngine, BacktestConfig


# Common index constituents (representative samples since full lists require data subscriptions)
# These are approximations based on publicly available information

RUSSELL_1000 = [
    # Top ~300 by market cap (representative)
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK.B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
    "DHR", "NEE", "VZ", "ADBE", "TXN", "PM", "CMCSA", "NKE", "RTX", "NFLX",
    "BMY", "HON", "UNP", "ORCL", "QCOM", "COP", "LOW", "UPS", "MS", "SPGI",
    "IBM", "INTC", "CAT", "BA", "GE", "SBUX", "DE", "INTU", "AMD", "AMGN",
    "GS", "BLK", "MDLZ", "ISRG", "ADP", "GILD", "ADI", "SYK", "C", "REGN",
    "T", "LMT", "PLD", "BKNG", "VRTX", "TMUS", "ZTS", "CI", "SCHW", "CVS",
    "CB", "SO", "MMC", "DUK", "MO", "PNC", "TJX", "CL", "BDX", "ITW",
    "USB", "TGT", "EOG", "SLB", "PSA", "CME", "AON", "NOC", "ICE", "WM",
    "FCX", "NSC", "FDX", "EMR", "GD", "APD", "MCK", "CCI", "D", "EW",
    "SRE", "GM", "F", "CARR", "PWR", "CBRE", "FICO", "WST", "AXON", "LYV",
    "EQR", "VTR", "DECK", "WAB", "IR", "VICI", "EXR", "RCL", "GWW", "DRI",
    "STE", "TER", "DOV", "ULTA", "ZBRA", "WAT", "BR", "BAX", "PKG", "EXPD",
    "DGX", "MKC", "TDY", "SWKS", "POOL", "SNA", "NTAP", "CF", "NVR", "AOS",
    "JBHT", "RJF", "TECH", "HWM", "TPR", "GNRC", "EPAM", "MPWR", "MANH", "PODD",
    # Add more to reach ~500
    "PAYC", "TYL", "ALLE", "HOLX", "MOH", "DPZ", "LKQ", "HAS", "HSIC", "RE",
    "CHRW", "IEX", "FFIV", "MKTX", "AIZ", "BIO", "FRT", "UDR", "CPT", "HST",
    "NI", "JKHY", "CBOE", "PNR", "ALB", "TXT", "BWA", "GL", "MAS", "SEE",
    "CMA", "HII", "ZION", "WRB", "ATO", "CFG", "FMC", "ETSY", "BEN", "LUMN",
    "NWL", "BBWI", "AAL", "CCL", "NCLH", "UAL", "DAL", "LUV", "ALK", "JBLU",
]

RUSSELL_2000_SAMPLE = [
    # Small-cap sample (~200 stocks)
    "APPS", "AMBA", "BLKB", "CAKE", "CARG", "CATY", "CEIX", "CHEF", "CHGG", "CIEN",
    "CLVS", "COOP", "CORT", "CPRX", "CROX", "CRUS", "CVBF", "DORM", "DXC", "EBIX",
    "EGHT", "ENPH", "ENTA", "ENV", "EQIX", "EVBG", "EWBC", "EXLS", "EXPO", "FARO",
    "FBP", "FCFS", "FHN", "FIVE", "FLO", "FOLD", "FORM", "FOXF", "FRPT", "FWRD",
    "GBCI", "GBX", "GDOT", "GEO", "GIII", "GLNG", "GMED", "GPOR", "GVA", "HA",
    "HAFC", "HBI", "HCI", "HELE", "HLF", "HMST", "HNI", "HOPE", "HPP", "HQY",
    "HTH", "HUBG", "ICUI", "IIVI", "INFN", "INGN", "INVA", "IOSP", "IRBT", "ITGR",
    "JBSS", "JCOM", "JWN", "KALU", "KAMN", "KBH", "KFY", "KLIC", "KMPR", "KMT",
    "KN", "KNSL", "KNX", "KRG", "KSS", "LANC", "LBRT", "LFUS", "LGND", "LII",
    "LIVN", "LKFN", "LNTH", "LOAN", "LOPE", "LPG", "LSCC", "LTC", "LXP", "M",
    "MANT", "MATX", "MC", "MCRI", "MD", "MEDP", "MEI", "MEOH", "MGEE", "MGRC",
    "MIDD", "MMS", "MORN", "MPW", "MRCY", "MSTR", "MTG", "MTH", "MTX", "MUR",
    "MUSA", "NATI", "NBHC", "NBTB", "NCBS", "NEO", "NEOG", "NMIH", "NNI", "NOVT",
    "NPO", "NSIT", "NSP", "NUS", "NVAX", "NWE", "OGS", "OII", "OMCL", "ONB",
    "OPCH", "ORA", "OSIS", "OTTR", "PARR", "PATK", "PAYO", "PCRX", "PDCE", "PDCO",
    "PEB", "PEBO", "PFS", "PGNY", "PINC", "PIPR", "PLMR", "PLUS", "PNFP", "PRAA",
    "PRGS", "PRVA", "PSTL", "PTCT", "PTEN", "PVH", "QGEN", "QTWO", "R", "RAMP",
    "RBBN", "RCM", "RDN", "REG", "REXR", "RGP", "RHP", "RLI", "RMBS", "RNG",
    "ROCK", "RPD", "RUN", "SABR", "SAFE", "SAGE", "SANM", "SBCF", "SBGI", "SCSC",
    "SEDG", "SEM", "SFBS", "SHEN", "SHOO", "SIG", "SITC", "SITE", "SKT", "SKYW",
]

RUSSELL_3000_SAMPLE = RUSSELL_1000 + RUSSELL_2000_SAMPLE  # Combined

SP500_SAMPLE = [
    # S&P 500 (approximation)
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK.B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
    "DHR", "NEE", "VZ", "ADBE", "TXN", "PM", "CMCSA", "NKE", "RTX", "NFLX",
    "BMY", "HON", "UNP", "ORCL", "QCOM", "COP", "LOW", "UPS", "MS", "SPGI",
    "IBM", "INTC", "CAT", "BA", "GE", "SBUX", "DE", "INTU", "AMD", "AMGN",
    "GS", "BLK", "MDLZ", "ISRG", "ADP", "GILD", "ADI", "SYK", "C", "REGN",
    "T", "LMT", "PLD", "BKNG", "VRTX", "TMUS", "ZTS", "CI", "SCHW", "CVS",
    "CB", "SO", "MMC", "DUK", "MO", "PNC", "TJX", "CL", "BDX", "ITW",
]

SP400_SAMPLE = [
    # S&P 400 Mid-cap (sample)
    "APA", "AR", "ATI", "AXTA", "BJ", "BRKR", "BWA", "CACI", "CACC", "CADE",
    "CALM", "CBSH", "CDAY", "CDK", "CFR", "CHE", "CLH", "CNK", "CNO", "CNX",
    "COLM", "COOP", "CPRI", "CRI", "CRUS", "CVLT", "CW", "CWK", "CXO", "DAN",
    "DCI", "DDS", "DEI", "DKS", "DLB", "DLR", "DNKN", "DORM", "DXC", "EAT",
    "EEFT", "EGP", "ENS", "ESGR", "ESRT", "EXEL", "EXP", "EXPE", "FANG", "FCNCA",
    "FHN", "FLO", "FNB", "FNDA", "G", "GATX", "GEF", "GEO", "GNTX", "GOLF",
    "GRA", "HAE", "HAS", "HBAN", "HGV", "HLI", "HNI", "HRB", "HRI", "HSIC",
    "HUN", "IAA", "IBKR", "ICUI", "IDCC", "IDA", "INGR", "IPGP", "ITT", "IVZ",
    "JBL", "JEF", "JHG", "JLL", "JW.A", "KAR", "KBR", "KEX", "KMT", "KNX",
    "KRC", "LAMR", "LANC", "LII", "LKFN", "LM", "LNTH", "LPG", "LSTR", "LW",
]

SP600_SAMPLE = [
    # S&P 600 Small-cap (sample)
    "AAON", "AAWW", "ABG", "ABM", "ABTX", "ACC", "ACLS", "AEIS", "AEGN", "AEO",
    "AGCO", "AGO", "AIN", "AIRC", "AIT", "AL", "ALGT", "AMED", "AMKR", "AMWD",
    "ANDE", "AOS", "APOG", "ASGN", "ASH", "ASGN", "AVA", "AVNS", "AX", "AXS",
    "AZZ", "B", "BANC", "BANR", "BCEI", "BCO", "BELFB", "BFS", "BHLB", "BIG",
    "BKE", "BKH", "BLKB", "BMI", "BOH", "BOOM", "BOOT", "BOX", "BRC", "BRKL",
    "BUSE", "BXC", "CAL", "CALM", "CAMT", "CASA", "CASS", "CATY", "CBT", "CCBG",
    "CCS", "CCSI", "CDNA", "CECE", "CEIX", "CENX", "CHEF", "CHX", "CIEN", "CIM",
    "CKH", "CLB", "CLW", "CMP", "CMPO", "CNA", "CNK", "CNO", "CNR", "CNS",
]

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


@dataclass
class UniverseDefinition:
    """Definition of a universe for testing."""
    name: str
    description: str
    symbols: List[str]
    category: str  # "large_cap", "mid_cap", "small_cap", "broad", "sector", "custom"


# Predefined universes for search
QUANT_UNIVERSES = {
    "russell_1000": UniverseDefinition(
        name="Russell 1000",
        description="Large-cap US stocks (~1000 largest)",
        symbols=RUSSELL_1000,
        category="large_cap",
    ),
    "russell_2000": UniverseDefinition(
        name="Russell 2000",
        description="Small-cap US stocks",
        symbols=RUSSELL_2000_SAMPLE,
        category="small_cap",
    ),
    "russell_3000": UniverseDefinition(
        name="Russell 3000",
        description="Broad US market (large + small cap)",
        symbols=RUSSELL_3000_SAMPLE,
        category="broad",
    ),
    "sp500": UniverseDefinition(
        name="S&P 500",
        description="500 largest US companies",
        symbols=SP500_SAMPLE,
        category="large_cap",
    ),
    "sp400": UniverseDefinition(
        name="S&P 400 MidCap",
        description="Mid-cap US stocks",
        symbols=SP400_SAMPLE,
        category="mid_cap",
    ),
    "sp600": UniverseDefinition(
        name="S&P 600 SmallCap",
        description="Small-cap US stocks",
        symbols=SP600_SAMPLE,
        category="small_cap",
    ),
    "nasdaq100": UniverseDefinition(
        name="NASDAQ 100",
        description="100 largest non-financial NASDAQ stocks",
        symbols=NASDAQ_100,
        category="large_cap",
    ),
}


class UniverseSearcher:
    """
    Searches for optimal universe for a given strategy.
    """

    def __init__(
        self,
        memory_dir: str = "./memory",
        cache_dir: str = "./data_cache",
    ):
        self.memory_dir = memory_dir
        self.cache_dir = cache_dir
        self.data_loader = AlpacaDataLoader(cache_dir=cache_dir)
        self.backtest_engine = BacktestEngine(BacktestConfig())

        # Cache for loaded data
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def _load_universe_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        use_dynamic: bool = True,
        min_volume: int = 100_000,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load data for a universe."""
        cache_key = f"{len(symbols)}_{start_date.date()}_{use_dynamic}"

        if cache_key in self._data_cache:
            return self._data_cache[cache_key], None

        end_date = end_date or datetime.now()

        if use_dynamic:
            # Use dynamic universe (point-in-time)
            config = DynamicUniverseConfig(
                min_avg_volume=min_volume,
                min_trading_days=60,
                rebalance_frequency=21,
                max_symbols=len(symbols),
                exclude_leveraged=True,
            )
            builder = DynamicUniverseBuilder(config, cache_dir=self.cache_dir)
            prices, mask = builder.build_universe_mask(
                start_date=start_date,
                end_date=end_date,
                candidate_symbols=symbols,
            )
            return prices, mask
        else:
            # Static universe (has survivorship bias)
            data = self.data_loader.get_bars_safe(
                symbols=symbols,
                timeframe="1Day",
                start=start_date,
                end=end_date,
                batch_size=100,
            )
            if data.empty:
                return pd.DataFrame(), None
            prices = data["close"].unstack(level=0)
            return prices, None

    def search(
        self,
        strategy_func: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        strategy_name: str,
        universes: Optional[List[str]] = None,
        start_date: datetime = datetime(2016, 1, 1),
        end_date: Optional[datetime] = None,
        use_dynamic: bool = True,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Search for optimal universe for a strategy.

        Args:
            strategy_func: Function that takes (prices, universe_mask) and returns signals
            strategy_name: Name of the strategy
            universes: List of universe keys to test (default: all)
            start_date: Backtest start date
            end_date: Backtest end date
            use_dynamic: Use dynamic (point-in-time) universe construction
            save_results: Save results to memory

        Returns:
            DataFrame with results for each universe
        """
        universes = universes or list(QUANT_UNIVERSES.keys())

        print("=" * 70)
        print(f"UNIVERSE SEARCH: {strategy_name}")
        print("=" * 70)
        print(f"Universes to test: {len(universes)}")
        print(f"Dynamic (point-in-time): {use_dynamic}")
        print("=" * 70)

        results = []

        for universe_key in universes:
            if universe_key not in QUANT_UNIVERSES:
                print(f"[WARN] Unknown universe: {universe_key}")
                continue

            universe = QUANT_UNIVERSES[universe_key]
            print(f"\n[Testing] {universe.name} ({len(universe.symbols)} symbols)...")

            try:
                # Load data
                prices, mask = self._load_universe_data(
                    symbols=universe.symbols,
                    start_date=start_date,
                    end_date=end_date,
                    use_dynamic=use_dynamic,
                )

                if prices.empty:
                    print(f"  [ERROR] No data loaded")
                    continue

                print(f"  Loaded {prices.shape[1]} symbols, {prices.shape[0]} days")

                # Generate signals
                if mask is not None:
                    signals = strategy_func(prices, mask)
                else:
                    signals = strategy_func(prices, None)

                # Run backtest
                benchmark = prices.get("SPY")
                backtest_results = self.backtest_engine.run(prices, signals, benchmark)

                result = {
                    "universe": universe.name,
                    "universe_key": universe_key,
                    "category": universe.category,
                    "num_symbols": prices.shape[1],
                    "dynamic": use_dynamic,
                    "annual_return": backtest_results.annual_return,
                    "sharpe_ratio": backtest_results.sharpe_ratio,
                    "sortino_ratio": backtest_results.sortino_ratio,
                    "max_drawdown": backtest_results.max_drawdown,
                    "win_rate": backtest_results.win_rate,
                    "num_trades": backtest_results.num_trades,
                    "profit_factor": backtest_results.profit_factor,
                    "correlation": backtest_results.correlation,
                }
                results.append(result)

                status = "PASS" if backtest_results.sharpe_ratio >= 0.5 else "FAIL"
                print(f"  Return: {backtest_results.annual_return:.1%} | "
                      f"Sharpe: {backtest_results.sharpe_ratio:.2f} | "
                      f"DD: {backtest_results.max_drawdown:.1%} | {status}")

            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

        df = pd.DataFrame(results)

        if save_results and not df.empty:
            self._save_results(df, strategy_name)

        return df

    def _save_results(self, df: pd.DataFrame, strategy_name: str):
        """Save universe search results."""
        os.makedirs(self.memory_dir, exist_ok=True)

        filename = f"universe_search_{strategy_name.replace(' ', '_').lower()}.json"
        filepath = os.path.join(self.memory_dir, filename)
        df.to_json(filepath, orient="records", indent=2)
        print(f"\n[Saved] Results to {filepath}")

    def print_summary(self, df: pd.DataFrame):
        """Print summary of universe search results."""
        if df.empty:
            print("No results to summarize.")
            return

        print("\n" + "=" * 70)
        print("UNIVERSE SEARCH SUMMARY")
        print("=" * 70)

        # Sort by Sharpe
        df_sorted = df.sort_values("sharpe_ratio", ascending=False)

        print("\nRANKING BY SHARPE RATIO:")
        print("-" * 70)
        for i, row in df_sorted.iterrows():
            status = "✓" if row["sharpe_ratio"] >= 0.5 else "✗"
            print(f"  {status} {row['universe']:20} Sharpe: {row['sharpe_ratio']:6.2f} | "
                  f"Return: {row['annual_return']:7.1%} | DD: {row['max_drawdown']:7.1%}")

        # By category
        print("\nBY CATEGORY (avg Sharpe):")
        print("-" * 70)
        by_category = df.groupby("category")["sharpe_ratio"].mean().sort_values(ascending=False)
        for cat, sharpe in by_category.items():
            print(f"  {cat:15} {sharpe:.2f}")

        # Best overall
        best = df_sorted.iloc[0]
        print(f"\nBEST UNIVERSE: {best['universe']}")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {best['annual_return']:.1%}")
        print(f"  Max Drawdown: {best['max_drawdown']:.1%}")
        print(f"  Num Trades: {best['num_trades']}")

        print("=" * 70)


def search_universes(
    strategy_func: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    strategy_name: str,
    universes: Optional[List[str]] = None,
    use_dynamic: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to search universes for a strategy.

    Args:
        strategy_func: Function(prices, mask) -> signals
        strategy_name: Name of strategy
        universes: List of universe keys (default: all)
        use_dynamic: Use point-in-time universe (default: True)

    Returns:
        DataFrame with results
    """
    searcher = UniverseSearcher()
    results = searcher.search(
        strategy_func=strategy_func,
        strategy_name=strategy_name,
        universes=universes,
        use_dynamic=use_dynamic,
    )
    searcher.print_summary(results)
    return results


if __name__ == "__main__":
    # Example: Test reversal strategy across universes
    def reversal_strategy(prices, mask):
        """Short-term reversal strategy."""
        lookback = 5
        hold = 5
        bottom_n = 10

        returns = prices.pct_change(lookback)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for i in range(lookback, len(prices), hold):
            end_idx = min(i + hold, len(prices))

            # Get valid symbols from mask
            if mask is not None:
                valid = mask.iloc[i][mask.iloc[i]].index.tolist()
            else:
                valid = prices.columns.tolist()

            if len(valid) < bottom_n:
                continue

            ret = returns.iloc[i][valid].dropna()
            if len(ret) >= bottom_n:
                losers = ret.nsmallest(bottom_n).index.tolist()
                weight = 1.0 / len(losers)
                for j in range(i, end_idx):
                    for sym in losers:
                        signals.iloc[j, signals.columns.get_loc(sym)] = weight

        return signals.fillna(0)

    results = search_universes(
        strategy_func=reversal_strategy,
        strategy_name="Short-term Reversal",
        universes=["russell_1000", "russell_2000", "sp500", "nasdaq100"],
        use_dynamic=True,
    )
