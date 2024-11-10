"""
data_utils.py

These might include helper functions for data transformations, cleaning, augmentations, or any operations used
repeatedly within the dataloading process.
"""

import os
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_channel_time_series(df, datetime_col, quantities_to_plot, title="Channel Time Series Data"):
    """
    Plot specified quantities over time for a given dataset.

    Args:
        df (pd.DataFrame): The dataset containing time series data.
        datetime_col (str): The name of the column containing datetime values.
        quantities_to_plot (list of str): List of columns in `data` to plot on the time series.
        title (str): Title for the plot. Default is "Channel Time Series Data".
    """
    # Convert to datetime and sort by date
    df = df.query("name_cc == 'Taylor Swift'").copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col)

    # Normalize specified quantities
    for quantity in quantities_to_plot:
        df[quantity] = df[quantity] - df[quantity].min()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=25)
    for quantity in quantities_to_plot:
        sns.lineplot(data=df, x=datetime_col, y=quantity, ax=ax, label=quantity, marker='.', linestyle='-')
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.5)
    plt.xlim(df[datetime_col].min(), df[datetime_col].max())
    plt.show()


def plot_category_distribution(df, columns, category, x_logs, y_logs, kind="hist", print_summary=False):
    """
    Plots the distribution of the columns in the DataFrame for a specific category.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        columns (list of str): List of columns to plot
        category (str): Name of the category
        x_logs (list of bool): List of boolean values indicating if the data will be log-transformed on the x-axis
        y_logs (list of bool): List of boolean values indicating if the y-axis will be log-scaled for each plot
        kind (str): Type of plot to use in {"violin", "hist", "boxplot", "kde", "boxenplot"}
        print_summary (bool): if True, prints the summary statistics of the columns
    """
    fig, axs = plt.subplots(1, len(columns), figsize=(8 * len(columns), 6))

    # If there's only one column, axs is not a list, so we make it iterable
    if len(columns) == 1:
        axs = [axs]

    for i, (col, x_log, y_log) in enumerate(zip(columns, x_logs, y_logs)):
        # Apply log transformation if specified
        data = np.log(df[col] + 1) if x_log else df[col]

        # Plot based on the selected kind
        if kind == "violin":
            sns.violinplot(x=data, fill=False, ax=axs[i], linewidth=1)
        elif kind == "hist":
            sns.histplot(data, bins=100, ax=axs[i])
        elif kind == "boxplot":
            sns.boxplot(x=data, ax=axs[i])
        elif kind == "kde":
            sns.kdeplot(data, ax=axs[i])
        elif kind == "boxenplot":
            sns.boxenplot(x=data, ax=axs[i])

        # Set plot titles and labels
        axs[i].set_title(f"Distribution of {col} for the {category} category", fontsize=25)

        if x_log:
            axs[i].set_xlabel(f"log({col} + 1)", fontsize=20)
        else:
            axs[i].set_xlabel(f"{col}", fontsize=20)
        axs[i].set_ylabel("Count", fontsize=20)
        axs[i].grid(True, alpha=0.5)

        # Apply log scale to the y-axis if specified
        if y_log:
            axs[i].set_yscale('log')
            axs[i].set_ylabel("log(Count)", fontsize=20)

    plt.tight_layout()
    plt.show()

    # Print summary statistics if required
    if print_summary:
        print(f"Summary statistics for the {columns}:")
        print(df[columns].describe())


def compare_distribution_across_categories(df, columns, categories, x_logs, y_logs, kind="hist"):
    """
    Plot the distribution of the columns for the given categories.

    Args:
        df (pd.DataFrame): The dataframe containing the data from all categories
        columns (list of str): The columns to plot
        categories (list of str): The categories to plot
        x_logs (list of bool): Whether to log-transform the x-axis data for each column
        y_logs (list of bool): Whether to log-scale the y-axis for each plot
        kind (str): The kind of plot to use in {"violin", "hist", "boxplot", "kde", "boxenplot"}
    """
    df = df.copy()

    # Filter for the selected categories
    df = df[df["category"].isin(categories)]

    # Create a plot for each column (i.e. for each feature)
    fig, axs = plt.subplots(1, len(columns), figsize=(len(columns) * 8, 6))

    # If there's only one column, axs is not a list, so we make it iterable
    if len(columns) == 1:
        axs = [axs]

    for i, (col, x_log, y_log) in enumerate(zip(columns, x_logs, y_logs)):
        # Apply log transformation if specified
        df[col] = np.log(df[col] + 1) if x_log else df[col]

        # Plot based on the selected kind
        if kind == "hist":
            sns.histplot(data=df, x=col, hue="category", bins=100, kde=True, ax=axs[i], alpha=0.3)
        elif kind == "violin":
            sns.violinplot(data=df, x="category", y=col, ax=axs[i])
        elif kind == "boxplot":
            sns.boxplot(data=df, x="category", hue="category", y=col, ax=axs[i])
        elif kind == "kde":
            sns.kdeplot(data=df, x=col, hue="category", ax=axs[i])
        elif kind == "boxenplot":
            sns.boxenplot(data=df, x="category", hue="category", y=col, ax=axs[i])
        else:
            raise ValueError("Invalid plot kind. Choose from {'violin', 'hist', 'boxplot', 'kde', 'boxenplot'}")

        # Set titles and labels
        axs[i].set_title(f"Distribution of {col} across {', '.join(categories)}", fontsize=20)
        axs[i].set_xlabel(f"log({col} + 1)" if x_log else col, fontsize=16)
        axs[i].set_ylabel("Count" if not y_log else "log(Count)", fontsize=16)

        # Apply y-axis log scale if specified
        if y_log:
            axs[i].set_yscale('log')

    plt.tight_layout()
    plt.show()


def cast_channels_df(df):
    """
    For a dataframe of channels, downcast numerical columns to lower precision types, convert date columns to datetime,
    and convert object columns to string type for memory optimization.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.

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

    # Convert the join_date column to datetime
    df["join_date"] = pd.to_datetime(df["join_date"], format='mixed', errors='coerce')

    # Convert the columns to string when type is object
    df = df.apply(lambda x: x.astype('string') if x.dtype == 'object' else x)

    return df


def cast_videos_df(df):
    """
    For a dataframe of videos, downcast numerical columns to lower precision types, convert date columns to datetime,
    and convert object columns to string type for memory optimization.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.

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

    # Convert the crawl_date and upload_date columns to datetime
    df["crawl_date"] = pd.to_datetime(df["crawl_date"], format='mixed', errors='coerce')
    df["upload_date"] = pd.to_datetime(df["upload_date"], format='mixed', errors='coerce')

    # Convert the columns to string when type is object
    df = df.apply(lambda x: x.astype('string') if x.dtype == 'object' else x)

    return df

def get_stats_on_channel_category(df, category_name, corr_method='spearman'):
    """
    Get basic statistics on YouTube channels in a certain category.

    Args:
        df (pd.DataFrame): Dataset containing the information of YouTube channels in a given category.
        category_name (str): Name of the category to analyze.
        corr_method (str): The method to compute the correlation matrix.

    Returns:
        df_stats (pd.DataFrame): DataFrame with statistical summaries for the channels in the category.
    """
    # Deep copy to not modify the initial data
    df = df.copy(deep=True)

    print(f"Displaying statistics to study the YouTube channels in the category: {category_name}", end='\n\n')

    print(f"The category {category_name} consists of {df.shape[0]} channels.", end='\n')

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
    print(f"Descriptive statistics for the {category_name} category:")
    print(stats)

    # Extract the year and the month from the datetime column
    df['year'] = df['join_date'].dt.year
    df['month'] = df['join_date'].dt.month

    # Plot histograms for numerical columns
    numerical_columns = df.select_dtypes(include=['integer', 'float']).columns

    for col in numerical_columns:
        plt.figure(figsize=(10, 7))
        sns.histplot(df[col], bins=30, color='blue', linewidth=0)
        plt.title(f"Histogram of {col} in the {category_name} category", fontsize=25)
        plt.xlabel(f"Values for {col}", fontsize=20)
        if col == 'videos_cc' or col == 'subscribers_cc':
            plt.yscale('log')
            plt.ylabel("Count (log)", fontsize=20)
        else:
            plt.yscale('linear')
            plt.ylabel("Count", fontsize=20)

        plt.show()

    # Plot the correlation matrix
    corr_matrix = df[numerical_columns].corr(method=corr_method)
    sns.heatmap(corr_matrix, annot=True)
    plt.title(f"Correlation matrix of numerical columns in the {category_name} category", fontsize=25)
    plt.show()

    return stats


def get_stats_on_video_category(df, category_name, corr_method='spearman'):
    """
    Get basic statistics on YouTube videos in a certain category.

    Args:
        df (pd.DataFrame): Dataset containing the information of YouTube videos in a given category.
        category_name (str): Name of the category to analyze.
        corr_method (str): The method to compute the correlation matrix.

    Returns:
        df_stats (pd.DataFrame): DataFrame with statistical summaries for the videos in the category.
    """
    # Deep copy to not modify the initial data
    df = df.copy(deep=True)

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
    print(f"Descriptive statistics for the {category_name} category:")
    print(stats)

    # Extract the year and the month from the datetime columns
    # df['crawl_year'] = df['crawl_date'].dt.year
    # df['crawl_month'] = df['crawl_date'].dt.month
    df['upload_year'] = df['upload_date'].dt.year
    df['upload_month'] = df['upload_date'].dt.month

    # Plot histograms for numerical columns
    numerical_columns = df.select_dtypes(include=['integer', 'float']).columns

    for col in numerical_columns:
        plt.figure(figsize=(10, 7))
        sns.histplot(df[col], bins=30, color='blue', linewidth=0)
        plt.title(f"Histogram of {col} in the {category_name} category")
        if col == 'like_count' or col == 'view_count':
            plt.xscale('log')
            plt.xlabel(f"Values for {col} (log)")
            plt.yscale('log')
            plt.ylabel("Count (log)")
        else:
            plt.xscale('linear')
            plt.xlabel(f"Values for {col}")
            plt.yscale('linear')
            plt.ylabel("Count")
        plt.show()

    # Plot the correlation matrix
    corr_matrix = df[numerical_columns].corr(method=corr_method)
    sns.heatmap(corr_matrix, annot=True)
    plt.title(f"Correlation matrix of numerical columns in the {category_name} category")
    plt.show()

    return stats