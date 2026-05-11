import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration constants
DEFAULT_INPUT_FILE = "./input/returns.pkl"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_Y_SPACING = 0.5
DEFAULT_FIGURE_SIZE = (10, 8)
DEFAULT_DPI = 300
DEFAULT_BW_ADJUST = 1.5
DEFAULT_GRIDSIZE = 200
DEFAULT_CLIP_RANGE = (-5, 5)
DEFAULT_ALPHA = 0.4
DEFAULT_COLOR_PALETTE = "viridis"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_returns(input_path: str) -> pd.DataFrame:
    """Load returns DataFrame from a pickle file.

    Args:
        input_path: Path to the pickle file containing returns data.

    Returns:
        DataFrame containing the returns data.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        pd.errors.PickleError: If the file cannot be read as a pickle.
        ValueError: If the loaded data is not a valid DataFrame.
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    try:
        logger.info(f"Loading returns data from: {input_path}")
        returns = pd.read_pickle(input_path)

        if not isinstance(returns, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(returns)}")

        if returns.empty:
            raise ValueError("Loaded DataFrame is empty")

        logger.info(f"Successfully loaded returns data with shape: {returns.shape}")
        return returns

    except pd.errors.PickleError as e:
        logger.error(f"Error reading pickle file {input_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {input_path}: {e}")
        raise

def calculate_statistics(returns: pd.DataFrame) -> List[Dict[str, str]]:
    """Calculate summary statistics for each stock.

    Args:
        returns: DataFrame containing stock returns.

    Returns:
        List of dictionaries containing statistics for each stock.
    """
    stats = []
    for stock in returns.columns:
        data = returns[stock].dropna()
        stats.append({
            "Stock": stock,
            "Mean": f"{data.mean():.3f}",
            "Std Dev": f"{data.std():.3f}",
            "90th-10th %ile": f"{(np.percentile(data, 90) - np.percentile(data, 10)):.3f}",
            "Skewness": f"{data.skew():.2f}"
        })
    logger.info(f"Calculated statistics for {len(stats)} stocks")
    return stats


def setup_plot_style() -> None:
    """Configure matplotlib and seaborn plotting style."""
    sns.set_style("whitegrid", {"axes.facecolor": "white"})
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })


def create_kde_plots(ax: plt.Axes, returns: pd.DataFrame, y_spacing: float,
                    colors: List, bw_adjust: float = DEFAULT_BW_ADJUST,
                    gridsize: int = DEFAULT_GRIDSIZE,
                    clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE,
                    alpha: float = DEFAULT_ALPHA) -> None:
    """Create KDE plots for each stock with ridge-style offset.

    Args:
        ax: Matplotlib axes object.
        returns: DataFrame containing stock returns.
        y_spacing: Vertical spacing between KDE curves.
        colors: List of colors for each stock.
        bw_adjust: Bandwidth adjustment for KDE.
        gridsize: Number of points in KDE grid.
        clip_range: Range to clip KDE values.
        alpha: Transparency level for fills.
    """
    stocks = returns.columns

    for i, stock in enumerate(stocks):
        data = returns[stock].dropna()

        if len(data) < 2:
            logger.warning(f"Insufficient data for {stock}, skipping")
            continue

        # Create temporary KDE plot to get data
        kde = sns.kdeplot(data, bw_adjust=bw_adjust, gridsize=gridsize,
                         clip=clip_range, cumulative=False, ax=ax)

        # Extract KDE data
        x, y = kde.lines[-1].get_data()
        y_shifted = y + i * y_spacing

        # Create filled area
        ax.fill_between(x, i * y_spacing, y_shifted,
                       color=colors[i], alpha=alpha, label=stock)

        # Remove the temporary line
        kde.lines[-1].remove()


def style_axes(ax: plt.Axes, returns: pd.DataFrame, y_spacing: float) -> None:
    """Apply styling to the plot axes.

    Args:
        ax: Matplotlib axes object.
        returns: DataFrame containing stock returns.
        y_spacing: Vertical spacing between KDE curves.
    """
    stocks = returns.columns
    num_stocks = len(stocks)

    # Set y-axis
    ax.set_yticks(np.arange(0, num_stocks * y_spacing, y_spacing))
    ax.set_yticklabels(stocks)

    # Set labels and title
    ax.set_xlabel("Returns")
    ax.set_title("Ridge Plot of Stock Returns Distribution", weight="bold")

    # Clean up appearance
    ax.grid(False)
    sns.despine(left=True, bottom=True, ax=ax)


def add_statistics_table(fig: plt.Figure, stats: List[Dict[str, str]]) -> None:
    """Add a statistics table below the main plot.

    Args:
        fig: Matplotlib figure object.
        stats: List of statistics dictionaries.
    """
    # Prepare table data
    col_labels = list(stats[0].keys())
    table_data = [[stat[col] for col in col_labels] for stat in stats]

    # Create table
    table = plt.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='bottom',
                     bbox=[0, -0.6, 1, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style table headers
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')


def save_plot(fig: plt.Figure, output_path: str, file_name: str,
             dpi: int = DEFAULT_DPI, show_plot: bool = True) -> str:
    """Save the plot to file and optionally display it.

    Args:
        fig: Matplotlib figure object.
        output_path: Directory to save the plot.
        file_name: Name of the output file.
        dpi: Resolution for saved image.
        show_plot: Whether to display the plot.

    Returns:
        Full path to the saved file.
    """
    # Ensure output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create full output path
    if not file_name.startswith(datetime.now().strftime('%Y-%m-%d')):
        file_name = f"{datetime.now().strftime('%Y-%m-%d')}_{file_name}"

    output_file = output_dir / file_name

    # Save the figure
    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    logger.info(f"Ridge plot saved to: {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return str(output_file)


def plot_ridge_plot(returns: pd.DataFrame, output_path: str, file_name: str,
                   y_spacing: float = DEFAULT_Y_SPACING,
                   figure_size: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
                   color_palette: str = DEFAULT_COLOR_PALETTE,
                   show_plot: bool = True) -> str:
    """Generate and save ridge plot of stock returns with statistics table.

    Args:
        returns: DataFrame containing stock returns data.
        output_path: Directory to save the output image.
        file_name: Name for the output file.
        y_spacing: Vertical spacing between KDE curves.
        figure_size: Size of the figure (width, height).
        color_palette: Color palette name for seaborn.
        show_plot: Whether to display the plot after saving.

    Returns:
        Path to the saved plot file.

    Raises:
        ValueError: If returns DataFrame is empty or invalid.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")

    stocks = returns.columns
    num_stocks = len(stocks)

    if num_stocks == 0:
        raise ValueError("No stock columns found in returns DataFrame")

    logger.info(f"Creating ridge plot for {num_stocks} stocks")

    # Calculate statistics
    stats = calculate_statistics(returns)

    # Setup plotting style
    setup_plot_style()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    # Get colors
    colors = sns.color_palette(color_palette, num_stocks)

    # Create KDE plots
    create_kde_plots(ax, returns, y_spacing, colors)

    # Style the axes
    style_axes(ax, returns, y_spacing)

    # Add statistics table
    add_statistics_table(fig, stats)

    # Adjust layout for table
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)

    # Save and display
    return save_plot(fig, output_path, file_name, show_plot=show_plot)

def main() -> int:
    """Main function to handle CLI execution.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    parser = argparse.ArgumentParser(
        description="Generate a ridge plot from stock returns data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s -i input/returns.pkl -o plots/ -n my_ridge_plot.png
  %(prog)s --input data.pkl --y_spacing 0.8 --no-show
  %(prog)s -v --color_palette Set2 --figure_size 12 10"""
    )

    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input pickle file containing returns DataFrame. Default: {DEFAULT_INPUT_FILE}"
    )

    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the ridge plot image. Default: {DEFAULT_OUTPUT_DIR}"
    )

    parser.add_argument(
        "-n", "--name",
        default=f"{datetime.now().strftime('%Y-%m-%d')}_ridge_plot.png",
        help="Name for output image file. Date prefix will be added if not present."
    )

    parser.add_argument(
        "-y", "--y_spacing",
        type=float,
        default=DEFAULT_Y_SPACING,
        help=f"Vertical spacing between KDE curves. Default: {DEFAULT_Y_SPACING}"
    )

    parser.add_argument(
        "--figure_size",
        nargs=2,
        type=int,
        default=DEFAULT_FIGURE_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help=f"Figure size in inches (width height). Default: {DEFAULT_FIGURE_SIZE[0]} {DEFAULT_FIGURE_SIZE[1]}"
    )

    parser.add_argument(
        "--color_palette",
        default=DEFAULT_COLOR_PALETTE,
        help=f"Seaborn color palette name. Default: {DEFAULT_COLOR_PALETTE}"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Resolution for saved image. Default: {DEFAULT_DPI}"
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot after saving"
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

    # Validate inputs
    if args.y_spacing <= 0:
        logger.error("Y-spacing must be positive")
        return 1

    if args.dpi <= 0:
        logger.error("DPI must be positive")
        return 1

    try:
        # Load data
        returns = load_returns(args.input)

        # Generate and save the ridge plot
        output_file = plot_ridge_plot(
            returns=returns,
            output_path=args.output,
            file_name=args.name,
            y_spacing=args.y_spacing,
            figure_size=tuple(args.figure_size),
            color_palette=args.color_palette,
            show_plot=not args.no_show
        )

        logger.info(f"Process completed successfully. Plot saved to: {output_file}")
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())