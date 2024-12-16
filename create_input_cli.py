
import pandas as pd
import yfinance as yf
import argparse
import os

def ensure_directories_exist():
    """Ensure the 'input' and 'output' directories exist."""
    for directory in ["input", "output"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def load_stock_list(stock_file):
    """Load stock symbols from a text file."""
    if not os.path.exists(stock_file):
        raise FileNotFoundError(f"Stock file not found: {stock_file}")
    with open(stock_file, "r") as file:
        stocks = [line.strip() for line in file.readlines() if line.strip()]
    if not stocks:
        raise ValueError("The stock file is empty or invalid.")
    return stocks

def fetch_and_save_returns(stock_file, output_file):
    """Fetch stock data, calculate returns, and save to a pickle file."""
    # Load stock symbols
    stocks = load_stock_list(stock_file)
    print(f"Fetching data for stocks: {', '.join(stocks)}")

    # Download historical stock data
    data = yf.download(stocks, period="5y")

    # Calculate daily returns
    returns = data['Adj Close'].pct_change().dropna()

    # Save returns as a pickle file
    returns.to_pickle(output_file)
    print(f"Saved returns data to: {output_file}")

if __name__ == "__main__":
    # CLI parser
    parser = argparse.ArgumentParser(description="Create input returns data from stock symbols.")
    parser.add_argument("-s", "--stocks", default="input/stocks.txt",
                        help="Path to the text file containing stock symbols (one per line).")
    parser.add_argument("-o", "--output", default="input/returns.pkl",
                        help="Path to save the output pickle file (default: input/returns.pkl).")
    args = parser.parse_args()

    # Ensure required directories exist
    ensure_directories_exist()

    # Fetch and save returns data
    try:
        fetch_and_save_returns(args.stocks, args.output)
    except Exception as e:
        print(f"Error: {e}")
