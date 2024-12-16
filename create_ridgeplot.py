import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from datetime import datetime

def load_returns(input_path):
    """Load returns DataFrame from a pickle file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    print(f"Loading returns data from: {input_path}")
    return pd.read_pickle(input_path)

def plot_ridge_plot(returns, output_path, file_name, y_spacing):
    """Generate and save ridge plot of stock returns with statistics table."""
    # Define number of stocks and vertical spacing
    stocks = returns.columns
    num_stocks = len(stocks)

    # Calculate statistics for table
    stats = []
    for stock in stocks:
        data = returns[stock].dropna()
        stats.append({
            "Stock": stock,
            "Mean": f"{data.mean():.2f}",
            "Std Dev": f"{data.std():.2f}",
            "Width": f"{(np.percentile(data, 90) - np.percentile(data, 10)):.2f}"
        })

    # Set Seaborn style
    sns.set(style="whitegrid", rc={"axes.facecolor": "white"})

    # Prepare figure with 2 parts: main KDE plot and table
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palette
    colors = sns.color_palette("viridis", num_stocks)

    # Plot KDE for each stock
    for i, stock in enumerate(stocks):
        data = returns[stock].dropna()
        kde = sns.kdeplot(data, bw_adjust=1.5, gridsize=200, clip=(-5, 5), cumulative=False)
        x, y = kde.lines[-1].get_data()
        y_shifted = y + i * y_spacing
        ax.fill_between(x, i * y_spacing, y_shifted, color=colors[i], alpha=0.4)
        kde.lines[-1].remove()

    # Aesthetics for the KDE plot
    ax.set_yticks(np.arange(0, num_stocks * y_spacing, y_spacing))
    ax.set_yticklabels(stocks, fontsize=10)
    ax.set_xlabel("Returns", fontsize=12)
    ax.set_title("Ridge Plot of Returns for Stocks", fontsize=16, weight="bold")

    # Remove grid and spines for a cleaner look
    ax.grid(False)
    sns.despine(left=True, bottom=True)

    # Add table below the plot
    table_data = [[s["Stock"], s["Mean"], s["Std Dev"], s["Width"]] for s in stats]
    col_labels = ["Stock", "Mean", "Std Dev", "Width"]

    table_ax = plt.table(cellText=table_data, colLabels=col_labels, cellLoc='center',
                         loc='bottom', bbox=[0, -0.5, 1, 0.3])  # Position table
    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(10)

    # Adjust layout to accommodate the table
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)

    # Save the figure
    output_file = os.path.join(output_path, file_name)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Ridge plot saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    # CLI parser
    parser = argparse.ArgumentParser(description="Generate a ridge plot from a returns dataframe.")
    parser.add_argument("-i", "--input", default="./input/returns.pkl", 
                        help="Path to the input pickle file containing the returns DataFrame.")
    parser.add_argument("-o", "--output", default="./output", 
                        help="Directory to save the ridge plot image.")
    parser.add_argument("-n", "--name", default=f"{datetime.now().strftime('%Y-%m-%d')}_ridge_plot.png", 
                        help="Name for output image file.")
    parser.add_argument("-y", "--y_spacing", type=float, default=0.5,
                        help="Vertical spacing between KDE curves (default: 0.5).")
    args = parser.parse_args()

    # Load data
    try:
        returns = load_returns(args.input)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # Generate and save the ridge plot
    try:
        plot_ridge_plot(returns, args.output, args.name, args.y_spacing)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)