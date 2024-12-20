"""
diversity_utils.py

These might include helper functions for data transformations, cleaning, augmentations, or any operations used
repeatedly within the dataloading process.
"""

import spacy
import pandas as pd
import numpy as np
import os
from wordcloud import WordCloud
from . import data_utils
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from typing import Dict
from tqdm import tqdm
import seaborn as sns
import zipfile


# Constants
RESOURCES_FOLDER = 'resources'

# Enable tqdm progress bar for pandas
tqdm.pandas()

nlp = spacy.load("en_core_web_sm")
nlp.disable_pipe('parser')
nlp.disable_pipe('senter')


def preprocess_text(text):
    '''
    Preprocess the data into a suitable format for NLP
    '''
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_common_theme(words):
    """
    Extracts the most representative word as the common theme based on semantic similarity.

    Args:
        words (list of str): A list of words to analyze.

    Returns:
        str: The word that best represents the common theme.
    """
    similarity = [[nlp(word1).similarity(nlp(word2)) for word2 in words] for word1 in words]
    word_scores = {word: sum(sim) for word, sim in zip(words, similarity)}
    
    return max(word_scores, key=word_scores.get)


def process_and_cluster_dataframe(df, text_column, num_clusters=100, max_features=10000, stop_words='english', min_df=5, max_df=0.7):
    """
    Processes and clusters the given DataFrame based on the specified text column.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str or list): The column name or list of column names to perform the analysis on.
        num_clusters (int): Number of clusters for KMeans. Default is 100.
        max_features (int): Maximum number of features for TF-IDF. Default is 10,000.
        stop_words (str or list): Stop words for TF-IDF vectorizer. Default is 'english'.
        min_df (int): Minimum document frequency for TF-IDF. Default is 5.
        max_df (float): Maximum document frequency (proportion) for TF-IDF. Default is 0.7.

    Returns:
        pd.DataFrame: The processed DataFrame with added clustering columns.
    """
    # Combine columns if a list is provided
    if isinstance(text_column, list):
        for col in text_column:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")
        df['Combined_Text'] = df[text_column].fillna('').apply(lambda row: ' '.join(row), axis=1)
        column_to_analyze = 'Combined_Text'
    elif isinstance(text_column, str):
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
        column_to_analyze = text_column
    else:
        raise ValueError("text_column must be a string or a list of strings.")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words, min_df=min_df, max_df=max_df)
    text_features = vectorizer.fit_transform(df[column_to_analyze])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(text_features)

    # Extract top words for each cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    # Assign names to clusters
    cluster_names = {}
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :1]]  # Top 1 word
        cluster_names[i] = ", ".join(top_words)  # Create a name from top words

    # Map cluster numbers to names and delete the original cluster column with a number
    df['Cluster Name'] = df['Cluster'].map(cluster_names)
    df.drop(columns=['Cluster'], inplace=True)

    return df


def get_clusters_from_df(df, text_column, num_clusters=100, max_features=10000, stop_words='english', min_df=5, max_df=0.7):
    """
    Processes and clusters the given DataFrame based on the specified text column.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str or list): The column name or list of column names to perform the analysis on.
        num_clusters (int): Number of clusters for KMeans. Default is 100.
        max_features (int): Maximum number of features for TF-IDF. Default is 10,000.
        stop_words (str or list): Stop words for TF-IDF vectorizer. Default is 'english'.
        min_df (int): Minimum document frequency for TF-IDF. Default is 5.
        max_df (float): Maximum document frequency (proportion) for TF-IDF. Default is 0.7.

    Returns:
        list: A list of cluster names for each row in the DataFrame.
    """
    # Combine columns if a list is provided
    if isinstance(text_column, list):
        for col in text_column:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")
        combined_text = df[text_column].fillna('').apply(lambda row: ' '.join(row), axis=1)
    elif isinstance(text_column, str):
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
        combined_text = df[text_column].fillna('')
    else:
        raise ValueError("text_column must be a string or a list of strings.")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words, min_df=min_df, max_df=max_df)
    try:
        text_features = vectorizer.fit_transform(combined_text)
    except ValueError as e:
        print(f"TF-IDF Vectorization failed: {e}")
        return ['error']

    # KMeans Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(text_features)

    # Extract top words for each cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    cluster_names = {}
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :1]]  # Top 1 word for each cluster
        cluster_names[i] = ", ".join(top_words)

    # Map cluster numbers to names
    cluster_name_list = [cluster_names[label] for label in cluster_labels]

    return cluster_name_list


def plot_word_frequency(df, column, title="Word Frequency Visualization", ax=None):
    """
    Creates and displays a word cloud based on word frequencies in a DataFrame column.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        column (str): The name of the column containing words.
        title (str): Title of the word cloud plot.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.

    """
    # Get the counts of each word in the column
    word_counts = Counter(df[column])   
    
    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        max_words=200,
        colormap="Reds"
    ).generate_from_frequencies(word_counts)
    
    # Plot the word cloud
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16)


def plot_wordclouds(dataframes_clusterized):
    """
    Plots wordclouds for clusters in given categories and saves the figure as a PDF.

    Args:
        dataframes_clusterized (dict): Dictionary where keys are category names and values are DataFrames.
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the wordclouds of clusters for categories
    for i, (key, df) in enumerate(dataframes_clusterized.items()):
        plot_word_frequency(
            df, 
            column='Cluster Name', 
            title=f"Wordcloud of the different contents in \nvideos within the {key.split('_')[-1]} category", 
            ax=axes[i]
        )
    plt.tight_layout()
    data_utils.save_plot('wordcloud_music_vs_entertainment_general', plt, overwrite=True)
    plt.show()


def unzip_glove():
    """
    Unzips the GloVe embeddings file.
    """
    # Path to the zip file
    zip_path = os.path.join(RESOURCES_FOLDER, 'glove.6B.zip')

    # Unzipping the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RESOURCES_FOLDER)


def load_glove_embeddings(file_path):
    """
    Loads GloVe embeddings into a dictionary.

    Args:
        file_path (str): Path to the GloVe embeddings file (e.g., glove.6B.100d.txt).

    Returns:
        dict: A dictionary mapping words to their embedding vectors.
    """
    embeddings = {}
    with open(os.path.join(RESOURCES_FOLDER, file_path), 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def tokenize_list(word_list, embeddings, embedding_dim=300):
    """
    Converts a list of words to their corresponding word embeddings.

    Args:
        word_list (list): A list of words.
        embeddings (dict): Pre-trained word embeddings dictionary.
        embedding_dim (int): Dimension of the embedding vectors.

    Returns:
        list: A list of word embeddings corresponding to the input words.
    """
    if not isinstance(word_list, list):
        raise ValueError("Input must be a list of words.")

    return [embeddings.get(word, np.zeros(embedding_dim)) for word in word_list]


def calculate_diversity_with_glove(df, embeddings):
    """
    Calculate diversity statistics for a DataFrame based on the identified clusters using GloVe embeddings.

    Args:
        df (pd.DataFrame): Processed DataFrame containing cluster names.
        embeddings (dict): GloVe embeddings as a dictionary mapping words to vectors.

    Returns:
        float: The average distance between clusters.
    """
    # Ensure cluster names exist
    if 'Cluster Name' not in df.columns:
        raise ValueError("'Cluster Name' column not found in the DataFrame.")

    # Extract unique cluster names
    cluster_names = df['Cluster Name'].unique()

    # Tokenize and embed cluster names
    embeddings_list = []
    for name in cluster_names:
        tokens = name.split()
        token_embeddings = [
            embeddings[token] for token in tokens if token in embeddings
        ]
        if token_embeddings:
            # Average embeddings for all tokens in the cluster name
            name_embedding = np.mean(token_embeddings, axis=0)
            embeddings_list.append(name_embedding)
        else:
            # If no valid embeddings found, skip the cluster name
            continue

    # Convert embeddings to numpy array
    if not embeddings_list:
        raise ValueError("No valid embeddings for cluster names in the DataFrame.")

    cluster_vectors = np.array(embeddings_list)

    # Calculate pairwise distances
    pairwise_distances = cosine_distances(cluster_vectors)

    # Calculate the average distance between clusters (exclude diagonal)
    if pairwise_distances.shape[0] == 1 and pairwise_distances.shape[1] == 1:
        return 0.0
    else:
        avg_distance = np.sum(pairwise_distances) / (pairwise_distances.shape[0] * (pairwise_distances.shape[1] - 1))

    return avg_distance

def calculate_diversity(embeddings_list):
    """
    Calculate diversity statistics based on a list of GloVe embeddings.

    Args:
        embeddings_list (list): List of embedding vectors.

    Returns:
        float: The average distance between embeddings in the list.
    """
    # Convert embeddings to numpy array
    if not embeddings_list:
        raise ValueError("The embeddings list is empty.")

    cluster_vectors = np.array(embeddings_list)

    # Calculate pairwise distances
    pairwise_distances = cosine_distances(cluster_vectors)

    # Calculate the average distance between embeddings (exclude diagonal)
    if pairwise_distances.shape[0] == 1 and pairwise_distances.shape[1] == 1:
        return 0.0
    else:
        avg_distance = np.sum(pairwise_distances) / (pairwise_distances.shape[0] * (pairwise_distances.shape[1] - 1))

    return avg_distance


def split_df_by_timeframe(df: pd.DataFrame, timeframe: str) -> Dict[str, pd.DataFrame]:
    """
    Splits the DataFrame into multiple DataFrames based on the specified timeframe.

    Parameters:
        df (pd.DataFrame): Input DataFrame with an 'upload_date' column as a datetime object.
        timeframe (str): Timeframe to split the DataFrame ('day', 'month', or 'year').

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are time intervals and values are DataFrames.
    """
    if timeframe not in {"day", "month", "year"}:
        raise ValueError("Timeframe must be one of: 'day', 'month', 'year'")

    if not pd.api.types.is_datetime64_any_dtype(df['upload_date']):
        raise TypeError("The 'upload_date' column must be a datetime object")

    # Define grouping logic based on timeframe
    if timeframe == "day":
        df["time_key"] = df["upload_date"].dt.strftime("%d-%m-%Y")
    elif timeframe == "month":
        df["time_key"] = df["upload_date"].dt.strftime("%m-%Y")
    elif timeframe == "year":
        df["time_key"] = df["upload_date"].dt.strftime("%Y")

    # Group by the 'time_key' column and split into DataFrames
    result = {time_key: group.drop(columns="time_key") for time_key, group in df.groupby("time_key")}

    return result


def plot_diversity(dfs, time_column='date', diversity_column='diversity_score', labels=None, title='Diversity Score Over Time', y_limit=None, filename=None, time_granularity='month'):
    """
    Plots the diversity score over time from multiple DataFrames.

    Parameters:
        dfs (list of pd.DataFrame): List of DataFrames, each with a datetime column and a diversity score column.
        time_column (str): Name of the column containing datetime objects.
        diversity_column (str): Name of the column containing diversity scores.
        labels (list of str): List of labels for the DataFrames to use in the legend.
        title (str): Title of the plot.
        y_limit (tuple of float, optional): Limits for the y-axis as (ymin, ymax).
        filename (str, optional): If provided, saves the plot to the specified file.
        time_granularity (str, optional): Granularity of the time axis (e.g., 'day', 'month', 'year'). Defaults to 'month'.

    Returns:
        None: Displays the plot or saves it to a file.
    """
    if labels is not None and len(labels) != len(dfs):
        raise ValueError("The number of labels must match the number of DataFrames.")

    plt.figure(figsize=(10, 9))

    for i, df in enumerate(dfs):
        # Check if the time column is a datetime object
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            raise TypeError(f"The column '{time_column}' in DataFrame {i} must be a datetime object.")

        df = df[df[diversity_column] != 0]  # Filter out rows where the diversity score is 0
        df = df.sort_values(by=time_column)
        label = labels[i] if labels else f"DataFrame {i+1}"
        plt.plot(df[time_column], df[diversity_column], linestyle='-', linewidth=2, label=label)

    plt.title(title)
    plt.xlabel(f"Upload Date ({time_granularity})")
    plt.ylabel("Diversity Score")
    plt.xticks(rotation=45, ha="right")
    if y_limit is not None:
        plt.ylim(y_limit)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    data_utils.save_plot(filename, plt, overwrite=True)
    plt.show()


def plot_diversity_histogram(diversities, filename='diversity_histogram', show_means=True):
    """
    Plots histograms of diversity values for multiple categories on the same plot.

    Args:
        diversities (dict): A dictionary where keys are category names and values are lists of diversity scores.
        filename (str): The file path to save the figure. If None, the figure will not be saved.
        show_means (bool): If True, show the mean of the diversity scores.
    """
    labels = ['Music', 'Entertainment']
    colors = sns.color_palette("tab10", len(diversities))
    plt.figure(figsize=(12, 8))

    for i, (category, values) in enumerate(diversities.items()):
        mean_value = np.mean(values)
        sns.histplot(values, bins=40, element="step", linewidth=3, log_scale=False, label=labels[i],
                      stats='density', alpha=0.2, fill=True, color=colors[i])
        if show_means:
            plt.axvline(mean_value, color=colors[i], linestyle="--" if i == 0 else "dotted",
                        label=f"Mean {labels[i].lower()}: {mean_value:.2f}", linewidth=3)

    plt.xlabel("Diversity Score")
    plt.ylabel("Number of Channels")
    plt.title("Distribution of the Diversity Scores by Category")
    plt.yscale("log")
    plt.legend(loc='best')
    plt.tight_layout()
    data_utils.save_plot(filename, plt, overwrite=True)
    plt.show()


def split_dataframes_by_timeframes(dataframes, timeframes):
    """
    Splits a dictionary of dataframes into sub-dictionaries based on specified timeframes.

    Args:
        dataframes (dict): A dictionary where keys are identifiers and values are dataframes.
        timeframes (list): A list of timeframes (e.g., ['day', 'month', 'year']).

    Returns:
        dict: A dictionary of dictionaries containing split dataframes for each timeframe.
    """
    split_dataframes = {timeframe: {} for timeframe in timeframes}

    for key, df in dataframes.items():
        for timeframe in timeframes:
            split_dataframes[timeframe][key] = split_df_by_timeframe(df, timeframe=timeframe)

    return split_dataframes


def compute_avg_distances(split_dataframes, embeddings, date_format='mixed'):
    """
    Computes the average distances between clusters for each timeframe and converts the results into dataframes.

    Args:
        split_dataframes (dict): A dictionary where keys are categories and values are dictionaries of dataframes split by timeframe.
        embeddings (any): Embeddings to use for the diversity calculation.
        date_format (str): The format to parse dates. Default is 'mixed'.

    Returns:
        dict: A dictionary of dataframes containing the average distances.
    """
    avg_distances_dataframes = {}

    for key, df_split in split_dataframes.items():
        avg_distances = {}
        for date, df in df_split.items():
            avg_distances[date] = calculate_diversity_with_glove(df, embeddings=embeddings)

        avg_distances_dataframes[key] = (
            pd.DataFrame.from_dict(avg_distances, orient='index')
            .reset_index()
            .rename(columns={'index': 'date', 0: 'diversity_score'})
        )
        avg_distances_dataframes[key]['date'] = pd.to_datetime(avg_distances_dataframes[key]['date'], format=date_format)

    return avg_distances_dataframes


def compute_channel_diversities(dataframes, channel_ids, glove_embeddings, topk=None):
    """
    Computes the average diversity for each channel in the given dataframes.

    Args:
        dataframes (dict): A dictionary of dataframes where keys are categories and values are the respective dataframes.
        channel_ids (dict): A dictionary where keys are categories and values are lists of channel IDs.
        glove_embeddings (any): Pre-trained GloVe embeddings for computing diversity.

    Returns:
        dict: A dictionary containing diversities for each channel in each category.
    """
    diversities = {}

    for key in channel_ids.keys():
        channel_ids[key] = channel_ids[key][:]  # Copy the list to ensure no modification
        if topk is not None:
            channel_ids[key] = channel_ids[key][:topk]

    for key, df in dataframes.items():
        diversities[key] = []
        for id in tqdm(channel_ids[key]):
            # Get the video metadata for the channel id
            video_metadata_for_id = df[df['channel_id'] == id]

            # Cluster the video metadata for the channel id
            video_metadata_for_id = get_clusters_from_df(
                video_metadata_for_id, 
                text_column=['title', 'tags'], 
                num_clusters=min(len(video_metadata_for_id), 25), 
                min_df=1, 
                max_df=1.0
            )

            # Get embeddings for the clusters
            embeddings = tokenize_list(video_metadata_for_id, glove_embeddings)

            # Compute the average distance between the clusters
            diversity = calculate_diversity(embeddings)
            diversities[key].append(diversity)

    return diversities