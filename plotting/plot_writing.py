import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import seaborn as sns
import wandb
from PyPDF2.generic import RectangleObject


def plot_combined_tldr_arxiv():
    """Plot TLDR and arXiv panels side by side with larger fonts."""

    # Initialize wandb API
    api = wandb.Api()

    # Project settings
    entity = "nu-llpr"
    project = "mlrl-archiv"

    # Get runs
    runs = api.runs(f"{entity}/{project}")

    # Separate runs for TLDR and arXiv
    tldr_runs = [run for run in runs if "tldr_level_reward" in run.name]
    arxiv_runs = [run for run in runs if "arxiv_level_reward" in run.name]

    print(f"Found {len(tldr_runs)} TLDR runs")
    print(f"Found {len(arxiv_runs)} arXiv runs")

    # Function to process runs data
    def process_runs_data(runs_list, max_step=1500):
        all_data = []
        for run in runs_list:
            try:
                history = run.history()

                # Use _step column and filter to reasonable range
                if "_step" in history.columns:
                    history = history[history["_step"] <= max_step]

                # Add run metadata
                history["run_name"] = run.name
                history["run_id"] = run.id

                all_data.append(history)

            except Exception as e:
                print(f"Error loading {run.name}: {e}")

        if not all_data:
            print("No data found")
            return None

        # Combine and clean data
        df = pd.concat(all_data, ignore_index=True)
        df = df[
            [
                "_step",
                "eval/avg_reward",
                "eval/details/level1_reward",
                "eval/details/level2_reward",
                "eval/details/level3_reward",
                "eval/details/transition_reward",
                "eval/details/jaccard_reward",
                "run_name",
                "run_id",
            ]
        ]
        df = df.dropna(
            subset=[
                "_step",
                "eval/avg_reward",
                "eval/details/level1_reward",
                "eval/details/level2_reward",
                "eval/details/level3_reward",
                "eval/details/transition_reward",
                "eval/details/jaccard_reward",
            ]
        )

        # Calculate structure reward as sum of level 1, 2 and 3 rewards
        df["structure_reward"] = (
            df["eval/details/level1_reward"]
            + df["eval/details/level2_reward"]
            + df["eval/details/level3_reward"]
        )

        return df

    # Process TLDR and arXiv data
    df_tldr = process_runs_data(tldr_runs)
    df_arxiv = process_runs_data(arxiv_runs)

    if df_tldr is None or df_arxiv is None:
        return

    # Function to create uniform data and aggregate
    def create_uniform_and_aggregate(df, max_step=1500, smooth=False, window_size=5):
        # Find minimum run length
        run_lengths = df.groupby("run_id").size()
        min_length = run_lengths.min()
        print(f"Run lengths: min={min_length}, max={run_lengths.max()}")

        # Create uniform x-axis mapping for each run
        uniform_data = []
        for run_id in df["run_id"].unique():
            run_data = df[df["run_id"] == run_id].copy()
            run_data = run_data.sort_values("_step")

            # Truncate to minimum length and create uniform x-axis
            run_data = run_data.head(min_length)
            run_data["uniform_step"] = np.linspace(0, max_step, len(run_data))

            uniform_data.append(run_data)

        # Combine uniform data
        df_uniform = pd.concat(uniform_data, ignore_index=True)

        # Aggregate data
        aggregated = (
            df_uniform.groupby("uniform_step")
            .agg(
                {
                    "eval/avg_reward": ["mean", "std"],
                    "structure_reward": ["mean", "std"],
                    "eval/details/transition_reward": ["mean", "std"],
                    "eval/details/jaccard_reward": ["mean", "std"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        aggregated.columns = [
            "uniform_step",
            "total_mean",
            "total_std",
            "structure_mean",
            "structure_std",
            "coherence_mean",
            "coherence_std",
            "consistency_mean",
            "consistency_std",
        ]

        # Apply smoothing if requested
        if smooth:
            # Smooth the mean values using rolling window
            aggregated["total_mean"] = (
                aggregated["total_mean"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            aggregated["structure_mean"] = (
                aggregated["structure_mean"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            aggregated["coherence_mean"] = (
                aggregated["coherence_mean"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            aggregated["consistency_mean"] = (
                aggregated["consistency_mean"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )

            # Also smooth the std values for consistency
            aggregated["total_std"] = (
                aggregated["total_std"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            aggregated["structure_std"] = (
                aggregated["structure_std"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            aggregated["coherence_std"] = (
                aggregated["coherence_std"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            aggregated["consistency_std"] = (
                aggregated["consistency_std"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )

        # Scale the aggregated means and stds
        # Total reward: 0-3 -> 0-1
        total_min, total_max = 0, 3
        aggregated["total_mean_scaled"] = (aggregated["total_mean"] - total_min) / (
            total_max - total_min
        )
        aggregated["total_std_scaled"] = aggregated["total_std"] / (
            total_max - total_min
        )

        # Structure reward: 0-2 -> 0-1
        struct_min, struct_max = 0, 2
        aggregated["structure_mean_scaled"] = (
            aggregated["structure_mean"] - struct_min
        ) / (struct_max - struct_min)
        aggregated["structure_std_scaled"] = aggregated["structure_std"] / (
            struct_max - struct_min
        )

        # Coherence reward: 0-0.4 -> 0-1
        coh_min, coh_max = 0, 0.4
        aggregated["coherence_mean_scaled"] = (
            aggregated["coherence_mean"] - coh_min
        ) / (coh_max - coh_min)
        aggregated["coherence_std_scaled"] = aggregated["coherence_std"] / (
            coh_max - coh_min
        )

        # Consistency reward: 0-0.6 -> 0-1
        cons_min, cons_max = 0, 0.6
        aggregated["consistency_mean_scaled"] = (
            aggregated["consistency_mean"] - cons_min
        ) / (cons_max - cons_min)
        aggregated["consistency_std_scaled"] = aggregated["consistency_std"] / (
            cons_max - cons_min
        )

        return aggregated

    # Process both datasets - smooth only arXiv
    aggregated_tldr = create_uniform_and_aggregate(df_tldr, smooth=True, window_size=2)
    aggregated_arxiv = create_uniform_and_aggregate(
        df_arxiv, smooth=True, window_size=5
    )

    # Create plot with two subplots
    plt.style.use("default")

    # Create figure with 2 subplots - made flatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))

    # Adjust spacing for flatter figure and make room for legend on right
    plt.subplots_adjust(left=0.07, right=0.83, wspace=0.35, bottom=0.20, top=0.90)

    # Define colors to match HumanEval style
    colors = {
        "Structure": "forestgreen",
        "Consistency": "indianred",
        "Coherence": "orange",
        "Total": "steelblue",
    }

    # Plot function with bounded error
    def plot_with_bounded_error(
        ax, x, y_mean, y_std, label, color, linestyle, alpha=1.0
    ):
        y_lower = np.maximum(y_mean - y_std, 0)  # Bound lower at 0
        y_upper = np.minimum(y_mean + y_std, 1)  # Bound upper at 1

        ax.plot(
            x,
            y_mean,
            label=label,
            linewidth=5,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)

    # INCREASED FONT SIZES (matching HumanEval)
    title_size = 24
    label_size = 24
    tick_size = 24

    # Plot TLDR (ax1)
    plot_with_bounded_error(
        ax1,
        aggregated_tldr["uniform_step"],
        aggregated_tldr["structure_mean_scaled"],
        aggregated_tldr["structure_std_scaled"],
        "Structure",
        colors["Structure"],
        "--",
        alpha=0.5,
    )

    plot_with_bounded_error(
        ax1,
        aggregated_tldr["uniform_step"],
        aggregated_tldr["consistency_mean_scaled"],
        aggregated_tldr["consistency_std_scaled"],
        "Consistency",
        colors["Consistency"],
        "--",
        alpha=0.5,
    )

    plot_with_bounded_error(
        ax1,
        aggregated_tldr["uniform_step"],
        aggregated_tldr["coherence_mean_scaled"],
        aggregated_tldr["coherence_std_scaled"],
        "Coherence",
        colors["Coherence"],
        "--",
        alpha=0.5,
    )

    plot_with_bounded_error(
        ax1,
        aggregated_tldr["uniform_step"],
        aggregated_tldr["total_mean_scaled"],
        aggregated_tldr["total_std_scaled"],
        "Total",
        colors["Total"],
        "-",
        alpha=1.0,
    )

    ax1.set_title("TLDR", fontsize=title_size, pad=10)
    ax1.set_xlabel("(K) Steps", fontsize=label_size)
    ax1.set_ylabel("Normalized Rewards", fontsize=label_size)
    ax1.set_xticks([0, 300, 600, 900, 1200, 1500])
    ax1.set_xticklabels(["0", "0.3", "0.6", "0.9", "1.2", "1.5"], fontsize=tick_size)
    ax1.yaxis.set_tick_params(labelsize=tick_size)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.1, 1.05)

    # Plot arXiv (ax2) - with smoothed data
    plot_with_bounded_error(
        ax2,
        aggregated_arxiv["uniform_step"],
        aggregated_arxiv["structure_mean_scaled"],
        aggregated_arxiv["structure_std_scaled"],
        "Structure",
        colors["Structure"],
        "--",
        alpha=0.6,
    )

    plot_with_bounded_error(
        ax2,
        aggregated_arxiv["uniform_step"],
        aggregated_arxiv["consistency_mean_scaled"],
        aggregated_arxiv["consistency_std_scaled"],
        "Consistency",
        colors["Consistency"],
        "--",
        alpha=0.6,
    )

    plot_with_bounded_error(
        ax2,
        aggregated_arxiv["uniform_step"],
        aggregated_arxiv["coherence_mean_scaled"],
        aggregated_arxiv["coherence_std_scaled"],
        "Coherence",
        colors["Coherence"],
        "--",
        alpha=0.6,
    )

    plot_with_bounded_error(
        ax2,
        aggregated_arxiv["uniform_step"],
        aggregated_arxiv["total_mean_scaled"],
        aggregated_arxiv["total_std_scaled"],
        "Total",
        colors["Total"],
        "-",
        alpha=1.0,
    )

    ax2.set_title("arXiv", fontsize=title_size, pad=10)
    ax2.set_xlabel("(K) Steps", fontsize=label_size)
    ax2.set_ylabel("Normalized Rewards", fontsize=label_size)
    ax2.set_xticks([0, 300, 600, 900, 1200, 1500])
    ax2.set_xticklabels(["0", "0.3", "0.6", "0.9", "1.2", "1.5"], fontsize=tick_size)
    ax2.yaxis.set_tick_params(labelsize=tick_size)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.1, 1.05)

    # Create custom legend handles with error bands
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    legend_elements = []

    # Create custom handles with line and surrounding error band
    for label, color in colors.items():
        # Determine line style based on label
        linestyle = "-" if label == "Total" else "--"
        alpha = 1.0 if label == "Total" else 0.6

        # Create a line for the legend
        line = Line2D(
            [0], [0], color=color, linewidth=5, linestyle=linestyle, alpha=alpha
        )

        # Create a rectangle to represent the error band
        rect = Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2, edgecolor="none")

        # Combine them into a single artist
        legend_elements.append((line, rect, label))

    # Custom legend handler to display both line and error band
    from matplotlib.legend_handler import HandlerBase

    class HandlerLineWithErrorBand(HandlerBase):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            line, rect, label = orig_handle

            # Create the error band rectangle
            rect_artist = Rectangle(
                (xdescent, ydescent),
                width,
                height,
                facecolor=rect.get_facecolor(),
                alpha=rect.get_alpha(),
                edgecolor="none",
                transform=trans,
            )

            # Create the line in the middle
            line_artist = Line2D(
                [xdescent, xdescent + width],
                [ydescent + height / 2, ydescent + height / 2],
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                alpha=line.get_alpha(),
                transform=trans,
            )

            return [rect_artist, line_artist]

    # Create the legend with custom handles
    legend_handles = [(h[0], h[1], h[2]) for h in legend_elements]
    legend_labels = [h[2] for h in legend_elements]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        fontsize=20,
        frameon=True,
        handlelength=2.2,
        title="Metrics",
        title_fontsize=24,
        handler_map={tuple: HandlerLineWithErrorBand()},
    )

    # Save the main plot WITH legend on the right
    timestamp = time.time()
    plt.savefig(f"./tldr_arxiv_all_{timestamp}.pdf", dpi=300)

    # Close the figure to free memory
    plt.close(fig)

    # Split the PDF with 2:3 ratio
    with open(f"./tldr_arxiv_all_{timestamp}.pdf", "rb") as file:
        reader = PyPDF2.PdfReader(file)
        left_writer = PyPDF2.PdfWriter()
        right_writer = PyPDF2.PdfWriter()

        for page in reader.pages:
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)
            split_point = page_width * 2 / 5  # 2:3 ratio split

            # Create left part (TLDR) - 2/5 of width
            left_page = PyPDF2.PageObject.create_blank_page(
                width=split_point, height=page_height
            )
            left_page.merge_page(page)
            left_page.mediabox = RectangleObject([0, 0, split_point, page_height])

            # Create right part (arXiv + legend) - 3/5 of width
            right_page = PyPDF2.PageObject.create_blank_page(
                width=page_width - split_point, height=page_height
            )
            page_copy = PyPDF2.PageObject.create_blank_page(
                width=page_width, height=page_height
            )
            page_copy.merge_page(page)
            page_copy.add_transformation(
                PyPDF2.Transformation().translate(-split_point, 0)
            )
            right_page.merge_page(page_copy)
            right_page.mediabox = RectangleObject(
                [0, 0, page_width - split_point, page_height]
            )

            left_writer.add_page(left_page)
            right_writer.add_page(right_page)

        # Save split PDFs
        with open(f"./tldr_{timestamp}.pdf", "wb") as left_file:
            left_writer.write(left_file)

        with open(f"./arxiv_{timestamp}.pdf", "wb") as right_file:
            right_writer.write(right_file)

    print(f"Created files:")
    print(f"- Main plot with legend: ./tldr_arxiv_all_{timestamp}.pdf")
    print(f"- TLDR half: ./tldr_{timestamp}.pdf")
    print(f"- arXiv half: ./arxiv_{timestamp}.pdf")

    return aggregated_tldr, aggregated_arxiv


# Run the plotting function
if __name__ == "__main__":
    plot_combined_tldr_arxiv()
