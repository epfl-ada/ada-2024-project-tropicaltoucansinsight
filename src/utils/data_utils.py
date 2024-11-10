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
    
    :param datasets: (list of tuples) A list where each tuple contains is a pair of (URL, filename).
    :param target_dir: (str) The directory where the downloaded files will be saved.
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



def get_stats_on_channel_category(data):
    """
    Get basic statistics on YouTube channels in a certain category.

    :param data (pd.DataFrame): Dataset containing the information of YouTube channels in a given category.
    :return (pd.DataFrame): DataFrame with statistical summaries (e.g., count, average views) for each channel category.
    """




def get_stats_on_video_category(data):
    """
    Get basic statistics on YouTube videos in a certain category.

    :param data (pd.DataFrame): Dataset containing the information of YouTube videos in a given category.
    :return (pd.DataFrame): DataFrame with statistical summaries (e.g., count, average views) for each video category.
    """
