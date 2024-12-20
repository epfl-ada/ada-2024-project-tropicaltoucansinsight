"""
monetization_utils.py

Helper functions to process the data analysis for monetization.
"""


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from . import data_utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# def compare_distrib(df_music, df_entertainment, to_compare):
#     """
#     Visualisation of the (normalized) distributions of a chosen feature
#     between two dataframes (usualy the 'Music' and the 'Entertainment' datasets)

#     Args:
#         df_music (pd.DataFrame): The dataframe containing the data from the 'Music' category
#         df_entertainment (pd.DataFrame): The dataframe containing the data from the 'Entertainment' category
#         to_compare (str): the name of the feature we want to compare
#     """

#     custom_labels = {
#         "duration": "Duration",
#         "estimated_revenue": "Estimated Revenue",
#         "view_count":"Number of Views",
#         "like_count":"Number of Likes",
#         "dislike_count":"Number of Dislikes",
#     }

#     sns.histplot(df_music[to_compare], label='Music', alpha=0.5, bins=50, log_scale=(True, False), stat='density')
#     sns.histplot(df_entertainment[to_compare], label='Entertainment', alpha=0.5, bins=50, log_scale=(True, False), stat='density')

#     plt.xlabel(to_compare)
#     plt.ylabel('Count')
#     plt.legend()
#     plt.show()

def compare_distrib(df_music, df_entertainment, to_compare, kind="hist", x_log=True, y_log=False, density=True, kde=True):
    """
    Visualisation of the distributions of chosen features between two dataframes
    (usually the 'Music' and 'Entertainment' datasets).

    Args:
        df_music (pd.DataFrame): The dataframe containing the data from the 'Music' category.
        df_entertainment (pd.DataFrame): The dataframe containing the data from the 'Entertainment' category.
        to_compare (str or list of str): The name(s) of the feature(s) to compare.
        kind (str): The type of plot to generate. Options are {'hist', 'violin', 'boxplot', 'kde'}.
        x_log (bool): Whether to apply a logarithmic transformation to the x-axis.
        y_log (bool): Whether to apply a logarithmic scale to the y-axis.
        density (bool): If True, normalize histograms to show densities.
    """
    # Convert single column to a list for unified processing
    if isinstance(to_compare, str):
        to_compare = [to_compare]

    # Custom labels for columns
    custom_labels = {
        "duration": "Duration",
        "estimated_revenue": "Estimated Revenue",
        "view_count": "Number of Views",
        "like_count": "Number of Likes",
        "dislike_count": "Number of Dislikes",
        "upload_month" : "Upload Month"
    }

    # Create subplots for each feature
    fig, axs = plt.subplots(len(to_compare), 1, figsize=(12, 6 * len(to_compare)), sharex=False)

    # If only one feature, axs is not a list
    if len(to_compare) == 1:
        axs = [axs]

    for i, feature in enumerate(to_compare):
        if x_log:
            # Avoid log(0) by adding 1
            df_music[feature] = df_music[feature] + 1
            df_entertainment[feature] = df_entertainment[feature] + 1

        if kind == "hist":
            sns.histplot(df_music[feature], label="Music", alpha=0.5, bins=80, kde=kde, log_scale=(x_log, y_log), stat="density" if density else "count", ax=axs[i])
            sns.histplot(df_entertainment[feature], label="Entertainment", alpha=0.5, bins=80, kde=kde, log_scale=(x_log, y_log), stat="density" if density else "count", ax=axs[i])

        # Customize labels and title
        axs[i].set_title(f"Distribution of {custom_labels.get(feature, feature)}", fontsize=16)
        axs[i].set_xlabel(custom_labels.get(feature, feature), fontsize=14)
        axs[i].set_ylabel("Density" if density else "Count", fontsize=14)
        axs[i].legend()

    plt.tight_layout()
    data_utils.save_plot(f"Distribution_of_{custom_labels.get(feature, feature)}", plt, overwrite=True)
    plt.show()


def keep_famous_videos(df_music, df_entertainment, threshold):
    """
    Creates two dataframes (one for the 'Music' category and one for the 'Entertainment' category)
    with only the most famous (viewed) videos

    Args:
        df_music (pd.DataFrame):         The dataframe containing the data from the 'Music' category
        df_entertainment (pd.DataFrame): The dataframe containing the data from the 'Entertainment' category
        threshold (float):               The minimum number of views of the videos we want to keep

    Returns:
        df_famous_music (pd.DataFrame):         The dataframe containing the most famous (views > threshold) from the 'Music' category
        df_famous_entertainment (pd.DataFrame): The dataframe containing the most famous (views > threshold) from the 'Entertainment' category
    """

    df_famous_music = df_music[df_music['view_count'] >= threshold].copy(deep=True)
    df_famous_entertainment = df_entertainment[df_entertainment['view_count'] >= threshold].copy(deep=True)

    return df_famous_music, df_famous_entertainment


def addapt_dataframes_for_prediction(public_df, df_music, df_entertainment, target):
    """
    Addapt the content of the 3 following types of dataframes :
    --> a dataframe from a dataset found on the web about a given YouTube channel, the 'Music' videos and 'Entertainment' dataframes
    so they all have their commun features at the end, and the public dataframe has the target feature for a future linear regression modelization

    Args:
        public_df (pd.DataFrame): The dataframe containing the reference data from the web
        df_music (pd.DataFrame): The dataframe containing the data from the 'Music' category
        df_entertainment (pd.DataFrame): The dataframe containing the data from the 'Entertainment' category
        target (str): the targeted feature of the public dataframe that we want to predict in a future linear regression modelization

    Returns:
        df_youtube_cleaned (pd.DataFrame): The dataframe from the web adapted to the commun format
        df_music_ToPredict (pd.DataFrame): The 'Music' category dataframe adapted to the commun format
        df_entertainment_ToPredict (pd.DataFrame): The 'Entertainment' category dataframe adapted to the commun format
        columns_to_keep (***): The list of left features in all the dataframes that represents this commun format
    """

    # creating ephemeral deep copies of our dataframes to be used solely for prediction purposes 
    df_music_ToPredict = df_music.copy(deep=True)
    df_entertainment_ToPredict = df_entertainment.copy(deep=True)
    df_youtube_cleaned = public_df.copy(deep=True)

    # df_music_ToPredict['upload_month'] = df_music_ToPredict['upload_date'].dt.month
    # df_entertainment_ToPredict['upload_month'] = df_entertainment_ToPredict['upload_date'].dt.month

    # Renaming the features with direct correspondancy wrt our dataframes
    rename_columns = {
        'ID': 'channel_id',
        'Video Duration': 'duration',
        'Video Publish Time': 'upload_date',
        'Views': 'view_count',
        'Likes': 'like_count',
        'Dislikes': 'dislike_count',
        'Month': 'upload_month'
    }
    df_youtube_cleaned.rename(columns=rename_columns, inplace=True)

    # Cleaning the public dataset
    df_youtube_cleaned = df_youtube_cleaned.dropna()
    # adapting the format of upload dates in the public dataframe
    df_youtube_cleaned["upload_date"] = pd.to_datetime(df_youtube_cleaned["upload_date"], format='mixed', errors='coerce')


    # Extraying, in our dataframes, the features of the public dataframe that we can easily determine 
    # day of month (in [0,30])
    df_music_ToPredict['Day'] = df_music_ToPredict['upload_date'].dt.day
    df_entertainment_ToPredict['Day'] = df_entertainment_ToPredict['upload_date'].dt.day
    # day of week (in [0,6])
    df_music_ToPredict['Day of Week'] = df_music_ToPredict['upload_date'].dt.dayofweek
    df_entertainment_ToPredict['Day of Week'] = df_entertainment_ToPredict['upload_date'].dt.dayofweek
    df_youtube_cleaned['Day of Week'] = df_youtube_cleaned['upload_date'].dt.dayofweek
    # Like rate
    epsilon = 1e-6
    df_music_ToPredict['Like Rate (%)'] = (df_music_ToPredict['like_count'] / (df_music_ToPredict['like_count'] + df_music_ToPredict['dislike_count'] + epsilon)) * 100
    df_entertainment_ToPredict['Like Rate (%)'] = (df_entertainment_ToPredict['like_count'] / (df_entertainment_ToPredict['like_count'] + df_entertainment_ToPredict['dislike_count'] + epsilon)) * 100


    # we filter:
    # keeping only the features that the dataframe from public data have in commun with our data, and the target feature we chose to predict 
    columns_to_keep = [
        'duration', 'upload_month', 'Day', 'Day of Week', 
        'Like Rate (%)', 'dislike_count', 'like_count', 'view_count'
    ]
    columns_for_model = columns_to_keep + [target]

    # Applying the filter
    df_music_ToPredict = df_music_ToPredict[columns_to_keep]
    df_entertainment_ToPredict = df_entertainment_ToPredict[columns_to_keep]
    df_youtube_cleaned = df_youtube_cleaned[columns_for_model]

    return df_youtube_cleaned, df_music_ToPredict, df_entertainment_ToPredict, columns_to_keep


def create_revenue_prediction_model_from_public_dataset(public_df, features, target):
    """
    Creates a Linear Regression model from the features of the public dataframe 
    in order to predict the 'target' feature of other dataframes 
    (usually of 'Music' and 'Entertainment' video metadata dataframes)

    Args:
        public_df (pd.DataFrame): The dataframe containing the reference data from the web
        features (list[str]): Different features used in the Linear Regression
        target (str): the targeted feature of the public dataframe that we want to predict with our linear regression model

    Returns:
        model: The final prediction model
    """


    # separating this features with our target vector
    X = public_df[features]
    y = public_df[target]

    # Dividing our reference data in train and test part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # normalizing the data since the features have different ranges and meanings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # creating and training the model
    # in our case it is a simple linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predicting on the test part
    y_predicted = model.predict(X_test)

    # Evaluating the model performances
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    r2 = r2_score(y_test, y_predicted)

    print("Linear Regression performance :")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² : {r2:.2f}")

    return model


def predict_revenue_from_public_dataset(public_df, df_music, df_entertainment, target):
    """
    Ads a 'Estimated Revenue (USD)' column to the 'Music' and 'Entertainment' video dataframes
    (Uses the prediction model to estimate the revenue of each video)

    Args:
        public_df (pd.DataFrame): The dataframe containing the reference data from the web
        df_music (pd.DataFrame): The dataframe containing the data from the 'Music' category
        df_entertainment (pd.DataFrame): The dataframe containing the data from the 'Entertainment' category
        target (str): the targeted feature of the public dataframe that we want to predict in a future linear regression modelization

    Returns:
        The two datasets ('Music' & 'Entertainment' usually) with their 'Estimated Revenue (USD)' column
    """

    df_youtube_cleaned, df_music_ToPredict, df_entertainment_ToPredict, features = addapt_dataframes_for_prediction(public_df, df_music, df_entertainment, target)

    model = create_revenue_prediction_model_from_public_dataset(df_youtube_cleaned, features, target)

    # Predicting revenues using our linear regression model
    df_music_ToPredict[target] = model.predict(df_music_ToPredict[features])
    df_entertainment_ToPredict[target] = model.predict(df_entertainment_ToPredict[features])

    return df_music_ToPredict, df_entertainment_ToPredict


def estimate_monetization(df_music, df_entertainment):

    """
    This function ads the 'Estimated Revenue (USD)' column, from the MP formula, to our dataframes

    Args:
        df_music (pd.DataFrame): The dataframe containing the data from the 'Music' category
        df_entertainment (pd.DataFrame): The dataframe containing the data from the 'Entertainment' category

    Return: none
    """

    # defining an hypothetic CPM for each category
    cpm_music = 1 # this one is smaller (see the reference)
    cpm_entertainment = 3

        # Calculating the estimated revenue for each category
    df_music['duration_factor'] = 1 + 1 * (df_music['duration'] // 480)  # factor for durations > 8 min
    df_music['Estimated Revenue (USD)'] = (
        (df_music['view_count'] / 1000) * cpm_music * df_music['duration_factor']
    )

    df_entertainment['duration_factor'] = 1 + 1 * (df_entertainment['duration'] // 480)  # same factor for durations > 8 min
    df_entertainment['Estimated Revenue (USD)'] = (
        (df_entertainment['view_count'] / 1000) * cpm_entertainment * df_entertainment['duration_factor']
    )