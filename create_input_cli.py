
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse

import pandas as pd
import yfinance as yf

# Configuration constants
DEFAULT_STOCK_FILE = "input/stocks.txt"
DEFAULT_OUTPUT_FILE = "input/returns.pkl"
DEFAULT_PERIOD = "5y"
REQUIRED_DIRECTORIES = ["input", "output"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directories_exist(directories: Optional[List[str]] = None) -> None:
    """Ensure required directories exist.

    Args:
        directories: List of directory paths to create. Defaults to REQUIRED_DIRECTORIES.

    Raises:
        OSError: If directory creation fails.
    """
    if directories is None:
        directories = REQUIRED_DIRECTORIES

    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

def load_stock_list(stock_file: str) -> List[str]:
    """Load stock symbols from a text file.

    Args:
        stock_file: Path to the text file containing stock symbols (one per line).

    Returns:
        List of stock symbols.

    Raises:
        FileNotFoundError: If the stock file doesn't exist.
        ValueError: If the stock file is empty or contains no valid symbols.
        IOError: If there's an error reading the file.
    """
    stock_path = Path(stock_file)

    if not stock_path.exists():
        raise FileNotFoundError(f"Stock file not found: {stock_file}")

    try:
        with open(stock_path, "r", encoding="utf-8") as file:
            stocks = [line.strip().upper() for line in file if line.strip()]
    except IOError as e:
        logger.error(f"Error reading stock file {stock_file}: {e}")
        raise

    if not stocks:
        raise ValueError(f"The stock file '{stock_file}' is empty or contains no valid symbols")

    logger.info(f"Loaded {len(stocks)} stock symbols from {stock_file}")
    return stocks

def fetch_and_save_returns(stock_file: str, output_file: str, period: str = DEFAULT_PERIOD) -> None:
    """Fetch stock data, calculate returns, and save to a pickle file.

    Args:
        stock_file: Path to the text file containing stock symbols.
        output_file: Path where the pickle file will be saved.
        period: Time period for historical data (default: 5y).

    Raises:
        FileNotFoundError: If the stock file doesn't exist.
        ValueError: If no valid data is retrieved or calculated.
        IOError: If there's an error saving the output file.
    """
    try:
        # Load stock symbols
        stocks = load_stock_list(stock_file)
        logger.info(f"Fetching data for {len(stocks)} stocks: {', '.join(stocks)}")

        # Download historical stock data
        logger.info(f"Downloading {period} of historical data...")
        data = yf.download(stocks, period=period, progress=False)

        if data.empty:
            raise ValueError("No data was retrieved from Yahoo Finance")

        # Handle single stock case (yfinance returns different structure)
        if len(stocks) == 1:
            adj_close = data['Adj Close']
        else:
            adj_close = data['Adj Close']

        # Calculate daily returns
        logger.info("Calculating daily returns...")
        returns = adj_close.pct_change().dropna()

        if returns.empty:
            raise ValueError("No valid returns could be calculated from the data")

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save returns as a pickle file
        returns.to_pickle(output_file)
        logger.info(f"Successfully saved returns data to: {output_file}")
        logger.info(f"Data shape: {returns.shape}")

    except Exception as e:
        logger.error(f"Error in fetch_and_save_returns: {e}")
        raise

def main() -> int:
    """Main function to handle CLI execution.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    parser = argparse.ArgumentParser(
        description="Create input returns data from stock symbols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s -s input/my_stocks.txt -o data/returns.pkl
  %(prog)s --stocks stocks.txt --output returns.pkl --period 2y"""
    )

    parser.add_argument(
        "-s", "--stocks",
        default=DEFAULT_STOCK_FILE,
        help=f"Path to the text file containing stock symbols (one per line). Default: {DEFAULT_STOCK_FILE}"
    )

    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to save the output pickle file. Default: {DEFAULT_OUTPUT_FILE}"
    )

    parser.add_argument(
        "-p", "--period",
        default=DEFAULT_PERIOD,
        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        help=f"Time period for historical data. Default: {DEFAULT_PERIOD}"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Ensure required directories exist
        ensure_directories_exist()

        # Fetch and save returns data
        fetch_and_save_returns(args.stocks, args.output, args.period)
        logger.info("Process completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
