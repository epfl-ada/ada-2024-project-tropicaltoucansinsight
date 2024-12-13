"""
data_utils.py

These might include helper functions for data transformations, cleaning, augmentations, or any operations used
repeatedly within the dataloading process.
"""
import re
import os
import gzip
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def get_data(datasets, target_dir="data"):
    """
    Download the specified datasets from the web and save them in the target directory.

    Args:
        datasets (list of tuples): A list where each tuple contains is a pair of (URL, filename).
        target_dir (str): The directory where the downloaded files will be saved.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    for dataset in datasets:
        # Ensure each dataset is a valid tuple
        if not isinstance(dataset, tuple) or len(dataset) != 2:
            print(f"Invalid dataset entry: {dataset}. It must be a tuple (URL, filename).")
            continue

        url, file_name = dataset
        file_path = os.path.join(target_dir, file_name)

        # Check if the file already exists, if not download it
        if not os.path.exists(file_path):
            print(f"Downloading {file_name} from {url}...")
            response = requests.get(url, stream=True)
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"{file_name} downloaded successfully.")

        else:
            print(f"{file_name} already exists in '{target_dir}'.")


def save_data_grouped_by_category(df, column, output_dir, overwrite=False, verbose=True):
    """
    Groups the channels by category and saves them to separate files.

    Args:
        df (pd.DataFrame): DataFrame containing the channel data.
        column (str): Column name to group the data by.
        output_dir (str): Output directory to save the files
        overwrite (bool): Whether to overwrite the files if they already exist.
        verbose (bool): Whether to print messages about the process.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_by_cat = df.groupby(column)
    for category, group_data in df_by_cat:
        output_file = os.path.join(output_dir, f"{category}.tsv.gz")
        if os.path.exists(output_file):
            if not overwrite:
                if verbose:
                    print(f"File '{output_file}' already exists. Skipping...")
                continue
            else:
                if verbose:
                    print(f"File '{output_file}' exists and will be overwritten.")
        group_data.to_csv(output_file, sep='\t', index=False, compression='gzip')


def get_channel_name(channel_id, df_channels):
    """
    Get the name of the channel given its id

    Args:
        channel_id (str): The channel id
        df_channels (pd.DataFrame): The DataFrame containing the channel information

    Returns:
        str: The name of the channel
    """
    return df_channels[df_channels["channel"] == channel_id]["name_cc"].values[0]


def merge_channel_name(df, df_channels, subscriber_rank=False):
    """
    Merge the channel id with the channel name

    Args:
        df (pd.DataFrame): The DataFrame to be merged
        df_channels (pd.DataFrame): The DataFrame containing the channel information
        subscriber_rank (bool): Whether to include the subscriber rank of the channel

    Returns:
        pd.DataFrame: The merged DataFrame
    """
    if subscriber_rank:
        df_merged = df.merge(df_channels[["channel", "name_cc", "subscriber_rank_sb"]], on="channel", how="left")
    else:
        df_merged = df.merge(df_channels[["channel", "name_cc"]], on="channel", how="left")
    return df_merged


def plot_channel_time_series(df, channel_name, datetime_col, quantities_to_plot, normalize=True,
                             palette="tab10", markers=["."], title="Channel Time Series Data"):
    """
    Plot specified quantities over time for a given dataset.

    Args:
        df (pd.DataFrame): The dataset containing time series data.
        datetime_col (str): The name of the column containing datetime values.
        quantities_to_plot (list of str): List of columns in `data` to plot on the time series.
        title (str): Title for the plot. Default is "Channel Time Series Data".
    """
    # Convert to datetime and sort by date
    df = df.query(f"name_cc == '{channel_name}'").copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col)

    # Normalize specified quantities
    if normalize:
        for quantity in quantities_to_plot:
            df[quantity] = (df[quantity] - df[quantity].min()) / (df[quantity].max() - df[quantity].min())

    custom_labels = {
        "views": "Views",
        "subs": "Subscribers",
        "delta_views": r"$\Delta$(views)",
        "delta_subs": r"$\Delta$(subs)"
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette(palette, len(quantities_to_plot))
    fig.suptitle(title, fontsize=25)
    for i, quantity in enumerate(quantities_to_plot):
        sns.lineplot(data=df, x=datetime_col, y=quantity, ax=ax, label=custom_labels.get(quantity, quantity), linestyle='-', markers=True,
                     marker=markers[i % len(markers)], color=colors[i % len(colors)], markeredgecolor='black',
                     markeredgewidth=0.3, linewidth=2)
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    if normalize:
        plt.ylabel("Normalized Count", fontsize=20)
    plt.legend(fontsize=20, bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid(True, alpha=0.5)
    plt.xlim(df[datetime_col].min(), df[datetime_col].max())
    plt.xticks(rotation=60)
    plt.show()


def plot_category_distribution(df_data, columns, category, x_logs, y_logs, kind="hist", print_summary=False):
    """
    Plots the distribution of the columns in the DataFrame for a specific category.

    Args:
        df_data (pd.DataFrame): DataFrame containing the data
        columns (list of str): List of columns to plot
        category (str): Name of the category
        x_logs (list of bool): List of boolean values indicating if the data will be log-transformed on the x-axis
        y_logs (list of bool): List of boolean values indicating if the y-axis will be log-scaled for each plot
        kind (str): Type of plot to use in {"violin", "hist", "boxplot", "kde", "boxenplot"}
        print_summary (bool): if True, prints the summary statistics of the columns
    """
    df = df_data.copy()

    custom_labels = {
        "views": "Views",
        "subs": "Subscribers",
        "delta_views": r"$\Delta$(views)",
        "delta_subs": r"$\Delta$(subs)",
        "delta_videos": r"$\Delta$(videos)",
        "view_count":"Number of Views",
        "like_count":"Number of Likes",
        "dislike_count":"Number of Dislikes",
        "collaborations_count": "Number of Collaborations",
        "colab_ratio": "Collaboration Ratio",
    }

    fig, axs = plt.subplots(len(columns), 1, figsize=(8, 6 * len(columns)))

    # If there's only one column, axs is not a list, so we make it iterable
    if len(columns) == 1:
        axs = [axs]

    for i, (col, x_log, y_log) in enumerate(zip(columns, x_logs, y_logs)):
        if x_log:
            df[col] = df[col] + 1  # Avoid log(0) by adding 1

        # Plot based on the selected kind
        if kind == "violin":
            sns.violinplot(x=df[col], fill=False, ax=axs[i], linewidth=1, label=custom_labels.get(col, col), log_scale=x_log)
        elif kind == "hist":
            sns.histplot(df[col], bins=100, ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log)
        elif kind == "boxplot":
            sns.boxplot(x=df[col], ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log)
        elif kind == "kde":
            sns.kdeplot(df[col], ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log)
        elif kind == "boxenplot":
            sns.boxenplot(x=df[col], ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log)
        else:
            raise ValueError("Invalid plot kind. Choose from {'violin', 'hist', 'boxplot', 'kde', 'boxenplot'}")

        # Set plot titles and labels
        axs[i].set_title(f"Distribution of Number of Entries per \n {custom_labels[col]} for the {category} category", fontsize=25)

        if x_log:
            axs[i].set_xlabel(fr"{custom_labels[col]}", fontsize=20)

        axs[i].set_ylabel("Count", fontsize=20)
        axs[i].grid(True, alpha=0.5)

        # Apply log scale to the y-axis if specified
        if y_log:
            axs[i].set_yscale('log')

    plt.tight_layout()
    plt.show()

    # Print summary statistics if required
    if print_summary:
        print(f"Summary statistics for the {columns}:")
        print(df[columns].describe())


def upper_case_first_letter(s):
    return s[0].upper() + s[1:]


def compare_distribution_across_categories(df_data, columns, categories, x_logs, y_logs, kind="hist", hue='category', marker_only=False, density=False):
    """
    Plot the distribution of the columns for the given categories.

    Args:
        df_data (pd.DataFrame): The dataframe containing the data from all categories
        columns (list of str): The columns to plot
        categories (list of str): The categories to plot
        x_logs (list of bool): Whether to log-transform the x-axis data for each column
        y_logs (list of bool): Whether to log-scale the y-axis for each plot
        kind (str): The kind of plot to use in {"violin", "hist", "boxplot", "kde", "boxenplot"}
        hue (str): The name of the column with the categories
        marker_only (bool): If True, only plot the markers without the histogram (useful when comparing many categories)
        density (bool): If True, plot the density instead of the count
    """
    # TODO: set the same color palette as the pie chart

    # Filter for the selected categories
    df = df_data[df_data[hue].isin(categories)].copy()
    df = df.rename(columns={hue: upper_case_first_letter(hue)})
    hue = upper_case_first_letter(hue)

    # Create a plot for each column (i.e. for each feature)
    fig, axs = plt.subplots(len(columns), 1, figsize=(12, 6 * len(columns)))

    custom_labels = {
        "views": "Views",
        "subs": "Subscribers",
        "videos": "Videos",
        "delta_views": r"$\Delta$(views)",
        "delta_subs": r"$\Delta$(subs)",
        "delta_videos": r"$\Delta$(videos)",
        "duration": "Duration",
        "engagement_score": "Engagement",
        "estimated_revenue": "Revenue",
        "view_count":"Number of Views",
        "like_count":"Number of Likes",
        "dislike_count":"Number of Dislikes",
        "collaborations_count": "Number of Collaborations",
        "colab_ratio": "Collaboration Ratio",
    }

    # If there's only one column, axs is not a list, so we make it iterable
    if len(columns) == 1:
        axs = [axs]

    for i, (col, x_log, y_log) in enumerate(zip(columns, x_logs, y_logs)):
        # Avoid log(0) by adding 1
        if x_log:
            df.loc[:, col] = df[col] + 1

        if kind == "hist":
            kde = True
            if y_log:
                kde = False
            if marker_only:
                markers = ['o', '.', '^', 'd', 'v', '<', '>', 'p', 'P', '*', 'h', 'H', '+', '|', '_', 's']
                if x_log:
                    bins = np.geomspace(df[col].min(), df[col].max(), 80)
                else:
                    bins = 80

                for cat, marker in zip(categories, markers):
                    data = df[df[hue] == cat][col]
                    counts, bin_edges = np.histogram(data, bins=bins, density=density)
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    axs[i].plot(bin_centers, counts, marker=marker, label=cat, alpha=0.9,
                                markerfacecolor='none', markeredgewidth=2, linestyle='None', markersize=7)
                    axs[i].legend(title=hue, title_fontsize=20)
                    axs[i].grid(True, alpha=0.5)
                if x_log:
                    axs[i].set_xscale('log')
            else:
                stat = 'density' if density else 'count'
                sns.histplot(data=df, x=col, hue=hue, bins=100, kde=kde, ax=axs[i], alpha=0.3, log_scale=x_log, stat=stat, common_norm=False)
                # Set titles and labels

            if density:
                axs[i].set_title(f"Density of {custom_labels[col]} across\n {', '.join(categories)}", fontsize=20)
                axs[i].set_ylabel("Density", fontsize=16)
            else:
                axs[i].set_title(f"Distribution of Number of Entries per {custom_labels[col]} across\n {', '.join(categories)}", fontsize=20)
                axs[i].set_ylabel("Count", fontsize=16)
            axs[i].set_xlabel(fr"{custom_labels[col]}", fontsize=16)

        elif kind == "violin":
            sns.violinplot(data=df, x=hue, hue=hue, y=col, ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log)
            axs[i].set_title(f"Violin plot of {custom_labels[col]} across\n {', '.join(categories)}", fontsize=20)
            axs[i].set_ylabel(fr"{custom_labels[col]}", fontsize=16)

        elif kind == "boxplot":
            sns.boxplot(data=df, x=hue, hue=hue, y=col, ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log)
            axs[i].set_title(f"Boxplot of {custom_labels[col]} across\n {', '.join(categories)}", fontsize=20)
            axs[i].set_ylabel(fr"{custom_labels[col]}", fontsize=16)

        elif kind == "kde":
            sns.kdeplot(data=df, x=col, hue=hue, ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log, common_norm=False)
            if density:
                axs[i].set_title(f"Density of {custom_labels[col]} across\n {', '.join(categories)}", fontsize=20)
                axs[i].set_ylabel("Density", fontsize=16)
            else:
                axs[i].set_title(f"Distribution of Number of Entries per {custom_labels[col]} across\n {', '.join(categories)}", fontsize=20)
                axs[i].set_ylabel("Count", fontsize=16)
            axs[i].set_xlabel(fr"{custom_labels[col]}", fontsize=16)

        elif kind == "boxenplot":
            sns.boxenplot(data=df, x=hue, hue=hue, y=col, ax=axs[i], label=custom_labels.get(col, col), log_scale=x_log)
            axs[i].set_title(f"Boxplot of {custom_labels[col]} across\n {', '.join(categories)}", fontsize=20)
            axs[i].set_ylabel(fr"{custom_labels[col]}", fontsize=16)

        else:
            raise ValueError("Invalid plot kind. Choose from {'violin', 'hist', 'boxplot', 'kde', 'boxenplot'}")

        # Apply y-axis log scale if specified
        if y_log:
            axs[i].set_yscale('log')

    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


def cast_df(df, type):
    """
    For a dataframe of channels or videos, downcast numerical columns to lower precision types, convert date columns to datetime,
    and convert object columns to string type for memory optimization.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        type (str): The type of the DataFrame to cast to (either 'channel' or 'video_metadata', 'time_series').

    Returns:
        new_df (pd.DataFrame): The new DataFrame with updated types.
    """
    # Deep copy to not modify the initial data
    df = df.copy(deep=True)

    # Downcast the numbers from 64 to 32 bits for less memory usage
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    if type == 'channel':
        # Convert the join_date column to datetime
        df["join_date"] = pd.to_datetime(df["join_date"], format='mixed', errors='coerce')

    elif type == 'video_metadata':
        # Convert the crawl_date and upload_date columns to datetime
        if 'crawl_date' in df.columns:
            df["crawl_date"] = pd.to_datetime(df["crawl_date"], format='mixed', errors='coerce')
        df["upload_date"] = pd.to_datetime(df["upload_date"], format='mixed', errors='coerce')

    elif type == 'time_series':
        # Convert the date column to datetime
        df["datetime"] = pd.to_datetime(df["datetime"], format='mixed', errors='coerce')

    # Convert the columns to string when type is object
    df = df.apply(lambda x: x.astype('string') if x.dtype == 'object' else x)

    return df


def get_stats_on_category(df, type, category_name, corr_method='spearman', verbose=True):
    """
    Get basic statistics on YouTube channels in a certain category.

    Args:
        df (pd.DataFrame): Dataset containing the information of YouTube channels in a given category.
        type (str): The type of the DataFrame to cast to (either 'channel' or 'video_metadata').
        category_name (str): Name of the category to analyze.
        corr_method (str): The method to compute the correlation matrix.

    Returns:
        df_stats (pd.DataFrame): DataFrame with statistical summaries for the channels in the category.
    """
    # Deep copy to not modify the initial data
    df = df.copy(deep=True)

    if type == 'channel':
        print(f"Displaying statistics to study the YouTube channels in the category: {category_name}", end='\n\n')
        print(f"The category {category_name} consists of {df.shape[0]} channels.", end='\n')

    elif type == 'video_metadata':
        print(f"Displaying statistics to study the YouTube videos in the category: {category_name}", end='\n\n')
        print(f"The category {category_name} consists of {df.shape[0]} videos.", end='\n')

    # Get the memory usage of the loaded dataframe
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (2 ** 20)  # As 1 MB = 2^20 bytes
    print(f"The DataFrame occupies {memory_mb:.2f} MB.", end='\n\n')

    # Get the count and percentage of missing values
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    missing_data = pd.DataFrame({
        'Missing values': missing_count,
        'Percentage missing': missing_percentage
    })

    # Get the descriptive statistics for numerical columns
    desc = df.describe().transpose()

    # Get the types of each column
    column_types = df.dtypes

    # Concatenate the descriptive stats, missing data stats, and column types
    stats = pd.concat([desc, missing_data, column_types.rename('Type')], axis=1)

    # Ensure the columns are in the right order: statistics followed by missing data stats and types
    stats = stats[
        ['Type', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'Missing values', 'Percentage missing']]

    # Print out the stats
    if verbose:
        print(f"Descriptive statistics for the {category_name} category:")
        print(stats)

    # Extract the year and the month from the datetime column
    if type == 'channel':
        df['year'] = df['join_date'].dt.year
        df['month'] = df['join_date'].dt.month
    elif type == 'video_metadata':
        df['upload_year'] = df['upload_date'].dt.year
        df['upload_month'] = df['upload_date'].dt.month

    # Plot histograms for numerical columns
    numerical_columns = df.select_dtypes(include=['integer', 'float']).columns

    custom_labels = {
        "videos_cc": "Number of Videos",
        "subscribers_cc": "Number of Subscribers",
        "like_count": "Number of Likes",
        "view_count": "Number of Views",
        "dislike_count": "Number of Dislikes",
        "duration": "Durations",
        "year": "Join Year",
        "month": "Join Month",
        "upload_year": "Upload Year",
        "upload_month": "Upload Month",
        "weights": "Weights",
        "subscriber_rank_sb": "Subscriber Rank",
        "join_date": "Join Date"
    }

    str_type = ''
    if type == 'channel': str_type = 'Channels'
    elif type == 'video_metadata': str_type = 'Videos'

    for col in numerical_columns:
        plt.figure(figsize=(8, 6))
        if (col == 'videos_cc' or col == 'subscribers_cc' or col == 'like_count' or
                col == 'view_count' or col == 'dislike_count' or col == 'duration'):
            sns.histplot(df[col], bins=100, log_scale=True)
            plt.title(f"Histogram of Number of {str_type} per \n {custom_labels[col]} in the {category_name} Category", fontsize=25)
            plt.xlabel(f"{custom_labels[col]}", fontsize=20)
            plt.ylabel("Count", fontsize=20)

            if col == 'dislike_count':  # TODO: voir si on garde plus tard ou pas
                plt.yscale('log')

        else:
            sns.histplot(df[col], bins=100)
            plt.xlabel(f"{custom_labels[col]}", fontsize=20)
            plt.ylabel("Count", fontsize=20)
            plt.title(f"Histogram of Number of {str_type} per \n {custom_labels[col]} in the {category_name} category", fontsize=25)



    plt.show()

    # Plot the correlation matrix
    corr_matrix = df[numerical_columns].corr(method=corr_method)
    sns.heatmap(corr_matrix, annot=True)
    plt.title(f"Correlation matrix of numerical columns \nin the {category_name} category", fontsize=25)
    plt.show()

    return stats


def save_chunk_grouped_by_col(df, column, output_dir="data/video_metadata"):
    """
    Saves a chunk of data into separate files for each category in the specified column.

    Args:
        df (pd.DataFrame): DataFrame containing the data chunk.
        column (str): Column to group the data by.
        output_dir (str): Output directory to save the files.
    """
    # Check if the column exists
    if column not in df.columns:
        raise ValueError(f"The column '{column}' does not exist in the DataFrame.")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group by the specified column
    grouped = df.groupby(column)

    for group, df_group in grouped:
        # Define the output file path
        output_file = os.path.join(output_dir, f"{group}.jsonl.gz")
        df_group.to_json(output_file, mode='a', compression='gzip', lines=True, orient='records')


def process_metadata(data_filename, output_dir='data/video_metadata', chunk_size=10_000, column_to_group="categories"):
    """
    Process a large JSONL file by reading it in chunks, grouping the data by a specified column, and saving each group to a separate Parquet file.

    Args:
        data_filename (str): Path to the data file
        chunk_size (int): The number of rows to read in each chunk
        column_to_group (str): The column to group the data by
    """
    with pd.read_json(data_filename, lines=True, compression='gzip', chunksize=chunk_size) as reader:
        for chunk in reader:
            # drop the "'crawl_date' column
            chunk = chunk.drop(columns=['crawl_date'])
            save_chunk_grouped_by_col(chunk, column=column_to_group, output_dir=output_dir)

    # Convert the JSONL files to Parquet format
    for filename in os.listdir(output_dir):
        if filename.endswith(".jsonl.gz"):
            data_filename = os.path.join(output_dir, filename)
            output_filename = data_filename.replace(".jsonl.gz", ".parquet")
            convert_jsonl_to_parquet(data_filename, output_filename)


def convert_jsonl_to_parquet(data_filename, output_filename, chunk_size=100_000):
    """
    Convert a JSONL file to Parquet format

    Args:
        data_filename (str): Path to the JSONL file
        output_filename (str): Path to the output Parquet file
    """
    dfs = []
    with pd.read_json(data_filename, lines=True, compression='gzip', chunksize=chunk_size) as reader:
        for chunk in reader:
            dfs.append(chunk)
    df = pd.concat(dfs)
    df.to_parquet(output_filename, compression='gzip')


def plot_pie_chart(df, column, title, values=None, threshold=3, palette="tab20"):
    """
    Plot a pie chart for the specified column in the DataFrame

    Args:
        df (pd.DataFrame): The DataFrame containing the data
        column (str): The column to plot
        title (str): The title of the pie chart
        values (str): The column containing the values to sum for each category
        threshold (int): The threshold percentage for displaying the labels outside the pie chart
    """
    custom_label = {
        "category_cc": "Category",
        "subscribers_cc": "Subscribers",
        "views_cc": "Views",
        "videos_cc": "Videos",
    }

    fig, ax = plt.subplots(figsize=(15, 10))
    # Group the data by the specified column and sum the values if specified, sort by descending order
    if values:
        data = df.groupby(column)[values].sum().sort_values(ascending=False)
    else:
        data = df[column].value_counts().sort_values(ascending=False)

    labels = data.index
    sizes = data.values

    # Custom colors for the pie chart: a category will have the same color across different plots
    unique_categories = df[column].unique()
    colors = sns.color_palette(palette, len(unique_categories))
    colors = [(r, g, b, 0.9) for r, g, b in colors]
    color_map = {category: color for category, color in zip(unique_categories, colors)}
    colors = [color_map[label] for label in labels]

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct=lambda p: f'{p:.1f}%' if p >= threshold else '',
        startangle=140,
        colors=colors,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        pctdistance=0.85
    )

    # Adjust percentages and add them outside or in legend for small segments
    legend_labels = []
    total = sum(sizes)
    for i, (size, label, autotext) in enumerate(zip(sizes, labels, autotexts)):
        percentage = (size / total) * 100
        # For small segments, add the percentage outside the pie chart
        if percentage < threshold:
            # Add to legend with percentage
            legend_labels.append(f"{label} ({percentage:.1f}%)")
        else:
            autotext.set_color('black')
            autotext.set_weight('bold')
            autotext.set_fontsize(14)
            legend_labels.append(label)

    ax.set_title(title, fontsize=20, weight='bold', pad=16)
    ax.legend(wedges, legend_labels, title=custom_label[column], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()


def geometric_mean(data):
    """
    Compute the geometric mean of a list of values

    Args:
        data (df.Series): List of values

    Returns:
        float: Geometric mean of the values
    """
    return np.exp(np.mean(np.log(data + 1)))


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


def preprocess_collaborations(chunk_df, collaboration_patterns=None):
    """
    Preprocess a chunk of data

    Args:
        chunk_df (pd.DataFrame): Chunk of data
        collaboration_patterns (list): List of regex patterns to detect collaborations.

    Returns:
        pd.DataFrame: Processed data
    """
    # Drop rows with missing values
    chunk_df = chunk_df.dropna(how='any')

    # Only keep Music and Entertainment categories with selected columns
    chunk_df = chunk_df[chunk_df['categories'].isin(['Music','Entertainment'])]
    columns_to_keep = ['categories', 'title', 'description', 'tags', 'view_count',
                       'like_count', 'dislike_count', 'channel_id', 'upload_date']
    chunk_df = chunk_df[columns_to_keep]

    # Only keep rows where the title indicates a collaboration
    chunk_df = chunk_df[chunk_df['title'].apply(detect_collaboration)]

    return chunk_df


def process_data(file_path, chunk_size, preprocess_func, output_path, collaboration_patterns=None):
    """
    Process a JSONL file in chunks and apply a preprocessing function to each chunk

    Args:
        file_path (str): Path to the gzipped JSONL file
        chunk_size (int): Number of rows to process per chunk
        preprocess_func (callable): Function to apply to each chunk of data (Pandas DataFrame)
        output_path (str): Path to store the processed data
        collaboration_patterns (list): List of regex patterns to detect collaborations

    Returns:
        None
    """
    with pd.read_json(file_path, lines=True, compression="gzip", chunksize=chunk_size) as reader:
        for chunk_df in reader:
            # Apply preprocessing function to the chunk
            processed_df = preprocess_func(chunk_df)

            # Append the processed chunk to the output file
            processed_df.to_json(output_path, orient="records", lines=True,
                                 force_ascii=False, compression='gzip', mode='a')


def process_video_counts(data_file, chunk_size, output_path='data/video_counts.jsonl.gz'):
    """
    Process a JSONL file containing video data to count the number of videos per channel

    Args:
        data_file (str): Path to the gzipped JSONL file
        chunk_size (int): Number of rows to process per chunk
        output_path (str): Path to store the processed data

    Returns:
        None
    """
    # Initialize a defaultdict to store the video counts per channel
    video_counts = defaultdict(int)

    # Process the data in chunks
    for i, chunk in enumerate(pd.read_json(data_file, chunksize=chunk_size, dtype={'channel_id': 'str'}, lines=True, compression='gzip')):
        counts = chunk['channel_id'].value_counts()
        for creator_id, count in counts.items():
            video_counts[creator_id] += count

    # Create a DataFrame from the video counts
    df_counts = pd.DataFrame(list(video_counts.items()), columns=['channel_id', 'video_count'])

    # Save the results to a compressed JSONL file
    print("Sauvegarde des résultats dans le fichier JSONL compressé...")
    df_counts.to_json(
        path_or_buf=output_path,
        orient='records',
        lines=True,
        compression='gzip',
        force_ascii=False
    )


def get_upload_evolution(df_music, df_entertainment, period=None, cumulative=True):
    """
    Plot the evolution of video uploads for "Music" and "Entertainment" categories over time

    Args:
        df_music (pandas.DataFrame): DataFrame containing upload dates for the "Music" category.
        df_entertainment (pandas.DataFrame): DataFrame containing upload dates for the "Entertainment" category.
        period (str, optional): Time period for aggregating uploads. Valid values are:
            - 'Y': Yearly
            - 'M': Monthly
            - 'W': Weekly
            - 'D': Daily
            If None, the function plots cumulative uploads over time
        cumulative (bool, optional): Whether to plot cumulative uploads over time. Ignored if `period` is provided
    """

    # Create a DataFrame with sorted upload dates for each category
    df_music_upload = pd.DataFrame(pd.to_datetime(df_music["upload_date"].copy()).sort_values(), columns=["upload_date"])
    df_music_upload["cumulative"] = range(1, len(df_music_upload) + 1)
    df_entertainment_upload = pd.DataFrame(pd.to_datetime(df_entertainment["upload_date"].copy()).sort_values(), columns=["upload_date"])
    df_entertainment_upload["cumulative"] = range(1, len(df_entertainment_upload) + 1)

    # If no period is given, plot the cumulative evolution
    if period == None and cumulative:
        x_min = df_entertainment_upload["upload_date"].min() if (df_entertainment_upload["upload_date"].min()
                                                                 < df_music_upload["upload_date"].min()) else df_music_upload["upload_date"].min()
        x_max = pd.Timestamp("2019-10")

        # Comparison between the 2 categories
        plt.figure(figsize=(20, 8))
        plt.plot(df_music_upload["upload_date"], df_music_upload["cumulative"], label="Music")
        plt.plot(df_entertainment_upload["upload_date"], df_entertainment_upload["cumulative"], label="Entertainment")
        plt.xlim(x_min, x_max)
        plt.xlabel("Upload Date")
        plt.ylabel("Cumulative Number of Uploads")
        plt.title("Cumulative Growth of Collaborative Video Uploads")
        plt.legend(fontsize=25)
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        valid_periods = {"Y", "M", "W", "D"}
        if period not in valid_periods:
            raise ValueError(f"Invalid period. Expected one of {valid_periods}, got '{period}' instead.")

        # Music
        df_music_period = df_music_upload.groupby(df_music_upload['upload_date'].dt.to_period(period)).size().reset_index(name='uploads')
        df_music_period["upload_date"] = df_music_period["upload_date"].dt.to_timestamp()

        # Entertainment
        df_entertainment_period = df_entertainment_upload.groupby(df_entertainment_upload['upload_date'].dt.to_period(period)).size().reset_index(name='uploads')
        df_entertainment_period["upload_date"] = df_entertainment_period["upload_date"].dt.to_timestamp()

        # Computing limits for the plot
        x_min = min(df_music_period["upload_date"].min(), df_entertainment_period["upload_date"].min())
        x_max = pd.Timestamp("2019-09")

        # Comparison between the two categories
        plt.figure(figsize=(20, 8))
        sns.lineplot(x='upload_date', y='uploads', data=df_music_period, label="Music")
        sns.lineplot(x='upload_date', y='uploads', data=df_entertainment_period, label="Entertainment")
        plt.xlim(x_min, x_max)
        period_label = {"M": "Month", "D": "Day", "Y": "Year", "W": "Week"}
        plt.xlabel(f"Upload Date ({period_label[period]})")
        plt.ylabel(f"Number of Uploads per {period_label[period]}")
        plt.title(f"Trend of Collaborative Video Uploads by {period_label[period]}")
        plt.legend(fontsize=25)
        plt.grid(True, alpha=0.3)
        plt.show()