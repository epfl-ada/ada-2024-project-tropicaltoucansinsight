"""
collab_utils.py

Helper functions to process the data analysis for collaborations.
"""


import re
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from . import data_utils
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from scipy.stats import mannwhitneyu, ks_2samp, cramervonmises_2samp


def process_views(file_path, output_path, chunk_size=500_000):
    """
    This function processes the views of the videos and keeps only the display_id, view_count and categories
    It also maps the categories to integers to save space: Music -> 0, Entertainment -> 1

    Args:
        file_path (str): Path to the file containing the metadata of the videos
        output_path (str): Path to the output file
        chunk_size (int): Number of lines to read at a time
    """
    # Map the categories to integers to save space
    category_map = {"Music": 0, "Entertainment": 1}

    # Process the data in chunks
    with pd.read_json(file_path, lines=True, compression="gzip", chunksize=chunk_size) as reader:
        for chunk in tqdm(reader):
            # Drop rows with missing values and map the categories to integers
            chunk = chunk.copy().dropna()
            chunk["categories"] = chunk["categories"].map(category_map)

            # Save the display_id, view_count and categories in a jsonl file
            columns = ["display_id", "view_count", "categories"]
            chunk[columns].to_json(output_path, orient="records", lines=True, force_ascii=False, compression='gzip',
                                   mode='a')


def top_p_views(df_views, p_threshold=0.9):
    """
    This function keeps only the videos that contribute to p% of the total views

    Args:
        df_views (pd.DataFrame): DataFrame containing the display_id and view_count of each video
        p_threshold (float): Proportion of the total views that we want to keep
    """
    # Sorting by the number of views and calculating the proportion of views
    total_views = df_views["view_count"].sum()
    df_views = df_views.sort_values(by="view_count", ascending=False).reset_index(drop=True)
    df_views["cumulative_proportion"] = df_views["view_count"].cumsum() / total_views

    # Keeping only the videos that contribute to p% of the total views
    df_views = df_views[df_views["cumulative_proportion"] <= p_threshold]

    # Calculating the proportion of views for each video
    df_views["proportion"] = df_views["view_count"] / total_views

    return df_views.copy()


def process_top_p_videos(file_path, output_path, top_p_videos):
    """
    This function processes the top p% of videos and keeps only the display_id and view_count

    Args:
        file_path (str): Path to the file containing the metadata of the videos
        output_path (str): Path to the output file
        top_p_videos (pd.DataFrame): DataFrame containing the display_id and view_count of the top p% of videos
    """
    # Read the metadata file in chunks
    with pd.read_json(file_path, lines=True, compression="gzip", chunksize=500_000) as reader:
        for chunk in tqdm(reader):
            # Merge the metadata with the top p% of videos
            chunk = chunk.merge(pd.DataFrame(top_p_videos["display_id"]), on="display_id", how="inner")
            columns_to_keep = ['categories', 'title', 'description', 'tags', 'view_count', 'like_count',
                               'dislike_count', 'channel_id', 'upload_date']

            # Save the results in a jsonl file
            chunk[columns_to_keep].to_json(output_path, orient="records", lines=True, force_ascii=False,
                                           compression='gzip', mode='a')


def collab_ratio(data_file, chunk_size=500000, output_path='data/collab_counts.jsonl.gz'):
    """
    This function processes the data to count the number of collaborations for each channel.
    It saves the results in a jsonl file with the channel_id and the number of collaborations

    Args:
        data_file (str): Path to the file containing the metadata of the videos
        chunk_size (int): Number of lines to read at a time
        output_path (str): Path to the output file

    Returns:
        None
    """
    # Initialize the dictionary to store the number of collaborations for each channel
    collab_count = defaultdict(int)

    # Process the data in chunks
    for i, chunk in enumerate(tqdm(pd.read_json(data_file,
                                                chunksize=chunk_size,
                                                dtype={'channel_id': 'str'}, lines=True,
                                                compression='gzip'), desc="Processing Chunks")):

        # Detect collaborations in the title of the videos
        chunk["collab"] = chunk["title"].apply(lambda x: data_utils.detect_collaboration(x))
        counts = chunk[chunk['collab'] == True]['channel_id'].value_counts()

        for creator_id, count in counts.items():
            collab_count[creator_id] += count

    # Save the results in a jsonl file
    df_counts = pd.DataFrame(list(collab_count.items()), columns=['channel_id', 'collab_count'])
    df_counts.to_json(path_or_buf=output_path, orient='records', lines=True, compression='gzip', force_ascii=False)


def top_p_results(df_views_music, df_top_p_music, df_views_entertainment, df_top_p_entertainment, p, plot=True, save=True):
    """
    Print the results for the top p% of videos

    Args:
        df_views_music (pd.DataFrame): DataFrame containing the views of the music videos
        df_top_p_music (pd.DataFrame): DataFrame containing the top p% of music videos
        df_views_entertainment (pd.DataFrame): DataFrame containing the views of the entertainment videos
        df_top_p_entertainment (pd.DataFrame): DataFrame containing the top p% of entertainment videos
        p (float): Proportion of the total views to keep
        plot (bool): If True, plot the results
    """
    category_data = [
        ("Music", len(df_views_music), len(df_top_p_music), len(df_top_p_music) / len(df_views_music)),
        ("Entertainment", len(df_views_entertainment), len(df_top_p_entertainment),
         len(df_top_p_entertainment) / len(df_views_entertainment))
    ]
    headers = ["Category", "Original Number of Videos", "Top Videos", "Fraction"]
    table_data = []
    for category, original_count, top_count, fraction in category_data:
        table_data.append([category, original_count, top_count, f"{fraction:.2%}"])
    print(f"Top {p * 100:.2f}%\n")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Compute the fraction of the total number of videos that the top p% represent
        fraction_music = len(df_top_p_music) / len(df_views_music)
        fraction_entertainment = len(df_top_p_entertainment) / len(df_views_entertainment)
        x_music = np.linspace(1 / len(df_views_music), fraction_music, len(df_top_p_music))
        x_entertainment = np.linspace(1 / len(df_views_entertainment), fraction_entertainment,
                                      len(df_top_p_entertainment))

        # Plot the cumulative fraction of views for the top p% of videos
        sns.lineplot(x=x_music, y=df_top_p_music["cumulative_proportion"], label="Music", ax=ax)
        sns.lineplot(x=x_entertainment, y=df_top_p_entertainment["cumulative_proportion"], label="Entertainment", ax=ax)
        ax.set_xlabel("Fraction of the Total Number of Videos")
        ax.set_ylabel("Cumulative Fraction of the Total Views")
        ax.set_title(rf"Cumulative Fraction of Views for the Top {p * 100:.0f}% of Videos", fontsize=25, pad=15)
        ax.legend()

        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.4)

        if save:
            data_utils.save_plot(f"top_{p}_views__Cumulative_Fraction.pdf", plt)

        plt.show()


def detect_collaboration(text, collaboration_patterns=None):
    """
    Detect if the text indicates a collaboration

    Args:
        text (str): Text to analyze
        collaboration_patterns (list): List of regex patterns to detect collaborations

    Returns:
        bool: True if collaboration is detected, False otherwise
    """
    # Default collaboration patterns
    if collaboration_patterns is None:
        collaboration_patterns = [r'\bfeat\b', r'\bft\b', r'\bfeaturing\b', r'\bx\b', r'\bw/\b', r'\bft\.\b']

    # Combine patterns into a single regex pattern
    pattern = re.compile('|'.join(collaboration_patterns), flags=re.IGNORECASE)

    return bool(pattern.search(text))


def print_collab_stats(top_p_music, top_p_entertainment):
    header = f"{'Category':<30} {'Number of Videos':<20} {'Number of Collaborations':<30} {'Fraction':<20}"
    print(header)
    print("-" * len(header))

    category_data = [
        ("Music", len(top_p_music), top_p_music["collab"].sum(), top_p_music["collab"].sum() / len(top_p_music)),
        ("Entertainment", len(top_p_entertainment), top_p_entertainment["collab"].sum(),
         top_p_entertainment["collab"].sum() / len(top_p_entertainment)),
    ]

    for category, total_videos, collab_count, fraction in category_data:
        print(f"{category:<30} {total_videos:<20} {collab_count:<30} {fraction:<20.2%}")


def plot_comparison_collab_and_non_collab(data, category, columns, x_logs, y_logs, custom_labels, save=True):
    """
    Plot the distribution of the features for collaborations and non-collaborations

    Args:
        data (pd.DataFrame): DataFrame containing the data
        category (str): Category of the videos
        columns (list): List of columns to plot
        x_logs (list): List of boolean values to set the x-axis to log scale
        y_logs (list): List of boolean values to set the y-axis to log scale
        custom_labels (dict): Dictionary containing the custom labels for the columns
        save (bool): If True, save the figures
    """
    if category not in ["Music", "Entertainment"]:
        raise ValueError("category must be either 'Music' or 'Entertainment'")

    fig, ax = plt.subplots(1, len(columns), figsize=(12 * len(columns), 8))
    for i, (col, x_log, y_log) in enumerate(zip(columns, x_logs, y_logs)):
        sns.histplot(data=data, x=col, hue="collab", hue_order=[False, True], bins=80, common_norm=False,
                     fill=True, stat="density", ax=ax[i], log_scale=x_log, legend=True)
        ax[i].set_xlabel(custom_labels[col])
        ax[i].set_ylabel("Normalized Number of Videos")
        ax[i].set_title(f"Distribution of the {custom_labels[col]}\n for {category} Videos")
        ax[i].legend(title="Collaboration", labels=["Yes", "No"])
        if y_log:
            ax[i].set_yscale("log")
    plt.tight_layout()
    if save:
        data_utils.save_plot("Collab_vs_NonCollab__Hist.pdf", plt)
    plt.show()

    # Boxplot
    fig, ax = plt.subplots(1, len(columns), figsize=(12 * len(columns), 8))
    for i, (col, x_log, y_log) in enumerate(zip(columns, x_logs, y_logs)):
        sns.boxplot(
            data=data, x="collab", hue="collab", y=col, ax=ax[i], legend=False, showmeans=True,
            meanprops={'marker': 'o', 'markeredgecolor': 'black', 'markersize': 20, 'markerfacecolor': 'red'},
            showfliers=False  # comment this line to show the outliers
        )
        ax[i].set_xlabel("Collaboration")
        ax[i].set_ylabel(custom_labels[col])
        ax[i].set_title(f"Boxplot of the {custom_labels[col]}\n for {category} Videos")
        ax[i].set_yscale("log")

    mean_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', markersize=20,
                         label='Means')
    fig.legend(handles=[mean_marker], loc='upper left', bbox_to_anchor=(1, 0.9))
    plt.tight_layout()
    if save:
        data_utils.save_plot("Collab_vs_NonCollab__Boxplot.pdf", plt)
    plt.show()

    # Complementary Cumulative Distribution Function (CCDF)
    fig, ax = plt.subplots(1, len(columns), figsize=(12 * len(columns), 8))
    for i, (col, x_log, y_log) in enumerate(zip(columns, x_logs, y_logs)):
        sns.ecdfplot(data=data, x=col, hue="collab", complementary=True, ax=ax[i], log_scale=x_log, legend=True)
        ax[i].set_xlabel(custom_labels[col])
        ax[i].set_ylabel("CCDF")
        ax[i].set_title(f"CCDF of the {custom_labels[col]}\n for {category} Videos")
        ax[i].set_yscale("log")
        ax[i].legend(title="Collaboration", labels=["Yes", "No"])
    plt.tight_layout()
    if save:
        data_utils.save_plot("Collab_vs_NonCollab__CCDF.pdf", plt)
    plt.show()


def compare_means_collab_non_collab(data, columns, custom_labels):
    """
    Compare the means of the features for collaborations and non-collaborations

    Args:
        data (pd.DataFrame): DataFrame containing the data
        columns (list): List of columns to compare
        custom_labels (dict): Dictionary containing the custom labels for the columns
    """
    # Define the column widths for the table
    rows = []

    # Compute the means for collaborations and non-collaborations
    for col in columns:
        metric_name = f"Mean {custom_labels[col]}"
        collab_mean = data[data['collab'] == True][col].mean()
        non_collab_mean = data[data['collab'] == False][col].mean()
        ratio = collab_mean / non_collab_mean

        # Append the row to the list
        rows.append([metric_name, f"{collab_mean:,.2f}", f"{non_collab_mean:,.2f}", f"{ratio:.2f}"])

    # Define the headers
    headers = ["Metric", "Collaborations", "Non-Collaborations", "Ratio (C/NC)"]

    # Print the table
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def test_distributions(data, columns, custom_labels):
    """
    Test if there is a significant difference between the distributions of collaborations and non-collaborations

    Args:
        data (pd.DataFrame): DataFrame containing the data
        columns (list): List of columns to compare
        custom_labels (dict): Dictionary containing the custom labels for the columns
    """
    rows = []

    # Perform statistical tests for each column
    for col in columns:
        data_collab = data[data["collab"] == True][col]
        data_non_collab = data[data["collab"] == False][col]
        label = custom_labels[col]

        # Mann-Whitney U test
        stat_u, pval_u = mannwhitneyu(data_collab, data_non_collab, alternative='two-sided')
        sig_u = "Yes" if pval_u < 0.05 else "No"
        rows.append([f"Mann-Whitney U ({label})", f"{stat_u:.2e}", f"{pval_u:.2e}", sig_u])

        # Kolmogorov-Smirnov test
        stat_ks, pval_ks = ks_2samp(data_collab, data_non_collab)
        sig_ks = "Yes" if pval_ks < 0.05 else "No"
        rows.append([f"Kolmogorov-Smirnov ({label})", f"{stat_ks:.2e}", f"{pval_ks:.2e}", sig_ks])

        # Cramér-von Mises test
        result = cramervonmises_2samp(data_collab, data_non_collab)
        stat_cv = result.statistic
        pval_cv = result.pvalue
        sig_cv = "Yes" if pval_cv < 0.05 else "No"
        rows.append([f"Cramér-von Mises ({label})", f"{stat_cv:.2f}", f"{pval_cv:.2f}", sig_cv])

    # Print the table
    headers = ["Test", "Statistic", "P-Value", "Significant?"]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def process_collab_count(data_file, chunk_size=500000, output_path='data/collab_counts.jsonl.gz'):
    """
    This function processes the data to count the number of collaborations for each channel.
    It saves the results in a jsonl file with the channel_id and the number of collaborations

    Args:
        data_file (str): Path to the file containing the metadata of the videos
        chunk_size (int): Number of lines to read at a time
        output_path (str): Path to the output file

    Returns:
        None
    """
    # Initialize the dictionary to store the number of collaborations for each channel
    collab_count = defaultdict(int)

    # Process the data in chunks
    for i, chunk in enumerate(tqdm(pd.read_json(data_file,
                                                chunksize=chunk_size,
                                                dtype={'channel_id': 'str'}, lines=True,
                                                compression='gzip'), desc="Processing Chunks")):

        # Detect collaborations in the title of the videos
        chunk["collab"] = chunk["title"].apply(lambda x: data_utils.detect_collaboration(x))
        counts = chunk[chunk['collab'] == True]['channel_id'].value_counts()

        for creator_id, count in counts.items():
            collab_count[creator_id] += count

    # Save the results in a jsonl file
    df_counts = pd.DataFrame(list(collab_count.items()), columns=['channel_id', 'collab_count'])
    df_counts.to_json(path_or_buf=output_path, orient='records', lines=True, compression='gzip', force_ascii=False)


def process_collab_ratio(collab_count_filepath, video_count_filepath, output_path='data/collab_ratio.jsonl.gz'):
    """
    Process the data to calculate the ratio of collaborations for each channel

    Args:
        collab_count_filepath (str): Path to the file containing the number of collaborations for each channel
        video_count_filepath (str): Path to the file containing the number of videos for each channel
        output_path (str): Path to the output file
    """
    # Load the data files
    df_video_count = pd.read_json(video_count_filepath, lines=True, compression="gzip")
    df_collab_count = pd.read_json(collab_count_filepath, lines=True, compression="gzip")

    # Merge the dataframes and calculate the collaboration ratio, some channels may not have collaborations
    # So we fill the NaN values with 0
    df_collab_ratio = df_video_count.merge(df_collab_count, on="channel_id", how="left")
    df_collab_ratio["collab_count"] = df_collab_ratio["collab_count"].fillna(0)
    df_collab_ratio["collab_ratio"] = df_collab_ratio["collab_count"] / df_collab_ratio["video_count"]

    # Save the results in a jsonl file
    df_collab_ratio.to_json(output_path, orient="records", lines=True, compression="gzip", force_ascii=False)


def plot_collab_ratio_distribution(df_music, df_entertainment, df_collab_ratio, p, save=True, show_means=True,
                                   kind="hist"):
    """
    Plot the distribution of the collaboration ratio for music and entertainment channels

    Args:
        df_music (pd.DataFrame): DataFrame containing the metadata of the music channels
        df_entertainment (pd.DataFrame): DataFrame containing the metadata of the entertainment channels
        df_collab_ratio (pd.DataFrame): DataFrame containing the collaboration ratio for each channel
        p (float): Proportion of the total views that we want to keep
        save (bool): If True, save the figure
        show_means (bool): If True, show the mean of the collaboration ratio
        kind (str): Type of plot to display: 'hist' or 'boxplot'
    """
    if kind not in ["hist", "boxplot"]:
        raise ValueError("kind must be either 'hist' or 'boxplot'")

    top_p_music = df_music.copy()
    top_p_entertainment = df_entertainment.copy()

    # Merge on the Music and Entertainment DataFrames
    top_p_music = top_p_music.merge(df_collab_ratio, on="channel_id", how="left")
    top_p_entertainment = top_p_entertainment.merge(df_collab_ratio, on="channel_id", how="left")

    music_grouped = top_p_music.groupby("channel_id")["collab_ratio"].mean().reset_index()
    entertainment_grouped = top_p_entertainment.groupby("channel_id")["collab_ratio"].mean().reset_index()

    mean_music = music_grouped["collab_ratio"].mean()
    mean_entertainment = entertainment_grouped["collab_ratio"].mean()

    music_color = "#1f77b4"
    entertainment_color = "#ff7f0e"
    if kind == "hist":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sns.histplot(data=music_grouped, x="collab_ratio", bins=80, ax=ax, log_scale=False, label="Music", alpha=0.2,
                     color=music_color, fill=True, linewidth=3, element="step")
        ax.axvline(mean_music, linestyle="--", label=f"Mean Music: {mean_music:.2f}", linewidth=3, color=music_color)
        sns.histplot(data=entertainment_grouped, x="collab_ratio", bins=80, ax=ax, log_scale=False,
                     label="Entertainment",
                     alpha=0.2, color=entertainment_color, fill=True, linewidth=3, element="step")
        ax.axvline(mean_entertainment, linestyle="dotted", label=f"Mean Entertainment: {mean_entertainment:.2f}",
                   linewidth=3, color=entertainment_color)
        ax.set_xlabel("Collaboration Ratio")
        ax.set_ylabel("Number of Channels")
        ax.set_title("Distribution of the Collaboration Ratio for Music Channels")
        ax.set_yscale("log")
        ax.legend(title="Category")
        plt.tight_layout()
        if save:
            data_utils.save_plot(f"top_{p}_views__Histplot.pdf", plt)
        plt.show()

    else:
        top_p_music["categories"] = "Music"
        top_p_entertainment["categories"] = "Entertainment"
        df = pd.concat([top_p_music, top_p_entertainment])
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sns.boxplot(
            x="categories", y="collab_ratio", data=df,
            hue="categories", ax=ax, showmeans=show_means,
            meanprops={'marker': 'o', 'markeredgecolor': 'black', 'markersize': 20, 'markerfacecolor': 'red'},
            showfliers=False, palette=[music_color, entertainment_color]
        )
        ax.set_xlabel("Category")
        ax.set_ylabel("Collaboration Ratio")
        ax.set_title(f"Distribution of the Collaboration Ratio\n for Music and Entertainment Channels")
        ax.set_yscale("log")

        if show_means:
            mean_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black',
                                 markersize=20, label='Means')
            ax.legend(handles=[mean_marker], loc='upper left', bbox_to_anchor=(1, 0.9))

        plt.tight_layout()
        if save:
            data_utils.save_plot(f"top_{p}_views__Boxplot.pdf", plt)
        plt.show()


def filter_channels_by_proportion(df_channels, p_threshold=0.9, top=True):
    """
    This function filters channels based on their contribution to a specified proportion of total subscribers.

    Args:
        df_channels (pd.DataFrame): DataFrame containing the channel_id and subscribers of each channel.
        p_threshold (float): Proportion of the total subscribers to consider (e.g., 0.9 for top 90%).
        top (bool): Whether to keep the top contributors (True) or the bottom contributors (False).

    Returns:
        pd.DataFrame: Filtered DataFrame of channels.
    """
    # Sorting by the number of subscribers and calculating the cumulative proportion
    total_subscribers = df_channels["subscribers_cc"].sum()
    df_channels = df_channels.sort_values(by="subscribers_cc", ascending=False).reset_index(drop=True)
    df_channels["cumulative_proportion"] = df_channels["subscribers_cc"].cumsum() / total_subscribers

    if top:
        # Keeping only channels contributing to p% of the total subscribers
        df_channels = df_channels[df_channels["cumulative_proportion"] <= p_threshold]
    else:
        # Keeping only channels contributing to (1-p)% of the total subscribers
        df_channels = df_channels[df_channels["cumulative_proportion"] > p_threshold]

    # Calculating the proportion of subscribers for each channel
    df_channels["proportion"] = df_channels["subscribers_cc"] / total_subscribers

    return df_channels.copy()


def print_top_channels_stats(df_music_channels, df_entertainment_channels, top_p_music_channels,
                             top_p_entertainment_channels, bottom_p_music_channels, bottom_p_entertainment_channels, p):
    """
    Print statistics about top and bottom channels for Music and Entertainment categories.

    Args:
        df_music_channels (pd.DataFrame): DataFrame containing all music channels
        df_entertainment_channels (pd.DataFrame): DataFrame containing all entertainment channels
        top_p_music_channels (pd.DataFrame): DataFrame containing top music channels
        top_p_entertainment_channels (pd.DataFrame): DataFrame containing top entertainment channels
        bottom_p_music_channels (pd.DataFrame): DataFrame containing bottom music channels
        bottom_p_entertainment_channels (pd.DataFrame): DataFrame containing bottom entertainment channels
        p (float): Proportion of channels to include in top group
    """
    rows = [
        [
            "Music",
            len(df_music_channels),
            len(top_p_music_channels),
            len(bottom_p_music_channels),
            f"{len(top_p_music_channels) / len(df_music_channels):.2%}",
            f"{len(bottom_p_music_channels) / len(df_music_channels):.2%}",
        ],
        [
            "Entertainment",
            len(df_entertainment_channels),
            len(top_p_entertainment_channels),
            len(bottom_p_entertainment_channels),
            f"{len(top_p_entertainment_channels) / len(df_entertainment_channels):.2%}",
            f"{len(bottom_p_entertainment_channels) / len(df_entertainment_channels):.2%}",
        ],
    ]

    headers = ["Category", "Original Number of Channels", "Top Channels", "Bottom Channels",
               "Top Fraction", "Bottom Fraction", ]

    print(f"Results for p = {p}:\n")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def compare_collab_ratio_top_p_channels(top_p_music_channels, bottom_p_music_channels,
                                        top_p_entertainment_channels, bottom_p_entertainment_channels,
                                        df_collab_ratio, p, save=True):
    """
    Compare the distribution of the collaboration ratio between the top p% of channels and the rest.

    Args:
        top_p_music_channels (pd.DataFrame): DataFrame containing the top p% of music channels
        bottom_p_music_channels (pd.DataFrame): DataFrame containing the bottom (1-p)% of music channels
        top_p_entertainment_channels (pd.DataFrame): DataFrame containing the top p% of entertainment channels
        bottom_p_entertainment_channels (pd.DataFrame): DataFrame containing the bottom (1-p)% of entertainment channels
        df_collab_ratio (pd.DataFrame): DataFrame containing the collaboration ratio for each channel
        p (float): Proportion of the total subscribers that we want to kept
        save (bool): If True, save the figure
    """
    # Compare the distribution of the collaboration ratio between the top p% of channels and the rest
    top_color = "#1f77b4"  # Blue
    bottom_color = "#d62728"  # Red

    top_mean_music = top_p_music_channels["collab_ratio"].mean()
    bottom_mean_music = bottom_p_music_channels["collab_ratio"].mean()
    top_mean_entertainment = top_p_entertainment_channels["collab_ratio"].mean()
    bottom_mean_entertainment = bottom_p_entertainment_channels["collab_ratio"].mean()

    fig, ax = plt.subplots(1, 2, figsize=(25, 8), sharey=True)

    sns.histplot(data=top_p_music_channels, x="collab_ratio", bins=80, ax=ax[0], element="step", linewidth=3,
                 log_scale=False, label="Top Music Channels", stat="density", alpha=0.3, fill=True, color=top_color)

    ax[0].axvline(top_mean_music, color=top_color, linestyle="--", label=rf"Top Mean = {top_mean_music * 100:.2f}%",
                  linewidth=3)

    sns.histplot(data=bottom_p_music_channels, x="collab_ratio", bins=80, ax=ax[0], element="step", linewidth=3,
                 log_scale=False, label="Bottom Music Channels", stat="density", alpha=0.15, fill=True, color=bottom_color)

    ax[0].axvline(bottom_mean_music, color=bottom_color, linestyle="-.",
                  label=rf"Bottom Mean = {bottom_mean_music * 100:.2f}%", linewidth=3)

    ax[0].set_xlabel("Collaboration Ratio")
    ax[0].set_ylabel("Normalized Number of Channels")
    ax[0].set_title("Distribution of the Collaboration Ratio for Music Channels")
    ax[0].legend()
    ax[0].set_yscale("log")

    sns.histplot(data=top_p_entertainment_channels, x="collab_ratio", bins=80, ax=ax[1], element="step", linewidth=3,
                 log_scale=False, label="Top Entertainment Channels", stat="density", alpha=0.3, fill=True, color=top_color)

    ax[1].axvline(top_mean_entertainment, color=top_color, linestyle="--",
                  label=rf"Top Mean = {top_mean_entertainment * 100:.2f}%", linewidth=3)

    sns.histplot(data=bottom_p_entertainment_channels, x="collab_ratio", bins=80, ax=ax[1], element="step", linewidth=3,
                 log_scale=False, label="Bottom Entertainment Channels", stat="density", alpha=0.15, fill=True,
                 color=bottom_color)

    ax[1].axvline(bottom_mean_entertainment, color=bottom_color, linestyle="-.",
                  label=rf"Bottom Mean = {bottom_mean_entertainment * 100:.2f}%", linewidth=3)
    ax[1].set_xlabel("Collaboration Ratio")
    ax[1].set_title("Distribution of the Collaboration Ratio for Entertainment Channels")
    ax[1].legend()
    ax[1].set_yscale("log")
    plt.tight_layout()

    if save:
        data_utils.save_plot(f"top_{p}_channels__Histplot.pdf", plt)

    plt.show()


def test_distribution_top_vs_bottom(top_data, bottom_data, categories, columns, alpha=0.05):
    """
    Perform statistical tests to compare the distributions of the top p% of channels and the rest.

    Args:
        top_data (list): List of DataFrames for the top p% of channels for each category.
        bottom_data (list): List of DataFrames for the bottom (1-p)% of channels for each category.
        categories (list): List of category names (e.g., ["Music", "Entertainment"]).
        columns (list): List of column names to test (e.g., ["collab_ratio"]).
    """
    rows = []
    for category, top_channels, bottom_channels in zip(categories, top_data, bottom_data):
        for col in columns:
            # Perform tests for each column
            for test in ["Mann-Whitney U", "Kolmogorov-Smirnov", "Cramér-von Mises"]:
                if test == "Mann-Whitney U":
                    stat, pval = mannwhitneyu(top_channels[col], bottom_channels[col], alternative='two-sided')
                elif test == "Kolmogorov-Smirnov":
                    stat, pval = ks_2samp(top_channels[col], bottom_channels[col])
                elif test == "Cramér-von Mises":
                    result = cramervonmises_2samp(top_channels[col], bottom_channels[col])
                    stat, pval = result.statistic, result.pvalue

                # Determine the significance
                sig = "Yes" if pval < alpha else "No"
                rows.append([f"{test} ({col})", category, f"{stat:.2e}", f"{pval:.2e}", sig])

    headers = ["Test", "Category", "Statistic", "P-Value", "Significant?"]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))