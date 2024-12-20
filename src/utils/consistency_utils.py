# This script contains all the utility functions for data analysis and visualization for Part 4: Popularity Consistency

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from . import data_utils


def clean_and_categorize(df_time_series):
    """
    Cleans the time-series DataFrame and separates it into Music and Entertainment caterogies

    Args:
        - df_time_series (pd.DataFrame): Time-series DataFrame
    
    Return:
        - df_time_series_music (pd.DataFrame): Music time-series DataFrame
        - df_time_series_entertainment (pd.DataFrame): Entertainment time-series DataFrame
    """

    # Drop rows with Nan entries
    df_time_series = df_time_series.dropna(axis='index')

    # Music category time-series DataFrame
    df_time_series_music = df_time_series.loc[df_time_series.category=='Music']

    # Entertainment category time-series DataFrame
    df_time_series_entertainment = df_time_series.loc[df_time_series.category=='Entertainment']

    return df_time_series_music, df_time_series_entertainment


def basic_stats(df_time_series, category):
    """
    Prints basic stats for the time-series DataFrame of a given category:
        - Number of entries
        - Number of entries where at least one video was published
        - Percentage of entries where at least one video was published

    Args: 
        - df_time_series (DataFrame): Time-series DataFrame of a given category
        - category (string): Specifies the category
    
    Return: None
    """

    # Print basic stats
    print(f"Number of entries for {category} category: {len(df_time_series)}")
    print(f"Number of entries for {category} with delta_videos >= 1: {len(df_time_series.loc[df_time_series.delta_videos>=1])}\
       ({len(df_time_series.loc[df_time_series.delta_videos>=1])/len(df_time_series)*100:.2f}% of total)")



def metric_dataframe(df, n_weeks, metric='delta_views'):
    """  
    TODO: Comment

    Args: 
        - df: Time-series DataFrame of interest from which the new DataFrame will be created
        - n_weeks: number of weeks over which we collect delta_views
        - metric: quantity of interest for which the DataFrame is created. Default is 'delta_views', but could also be delta_subs for example.

    Return: DataFrame as described above 
    """
    df_one_new_video = df[df['delta_videos'] == 1]
    df_list = []
    print(f"Number of initial video entries for n_weeks = {n_weeks}: {len(df_one_new_video)}")

    # Filter channel groups once for optimization
    grouped_by_channel = df.groupby('channel')

    #counter = 0

    for start_index in df_one_new_video.index:
        # Pre-filtered group avoids redundant channel filtering
        channel_data = grouped_by_channel.get_group(df.at[start_index, 'channel'])
        start_pos = channel_data.index.get_loc(start_index)

        # Verify timeframe and collect delta_views
        if start_pos + n_weeks < len(channel_data):
            df_n_weeks_after = channel_data.iloc[start_pos + 1 : start_pos + 1 + n_weeks]
            if df_n_weeks_after['delta_videos'].sum() < 1:
                metric_values = df_n_weeks_after[metric].values
                df_list.append({f"{i + 1}": metric_values[i] for i in range(len(metric_values))})

    # Convert collected data into a DataFrame
    result_df = pd.DataFrame(df_list)
    print(f"Number of valid entries for n_weeks = {n_weeks}: {len(result_df)}")

    return result_df


def plot_metric(df_music_metric_values, df_entertainment_metric_values, ax, first, median=False):
    """  
    Plots the mean time evolution (with standard deviation) on one Ax of a subplot for Music and Entertainment dataframes given as arguments, 
    for a given number of weeks.
    
    Args:
        - df_music_metric_values: Music DataFrame with values of Metric of interest
        - df_entertainment_metric_values: Entertainment with values of Metric of interest
        - ax (plt.Axes): ax specifying the subplot
        - first (boolean): specifies whether the ax is first in the row (column-wise)
        - median (boolean): if True plots the median for Music and Entertainment, if False doesn't plot
    
    Return: None (only plots)
    """

    # Music category mean and standard
    df_music_applied = df_music_metric_values.apply([np.mean, np.std, np.median], axis=0)
    df_music_applied.loc['std'] = df_music_applied.loc['std'].multiply(1/np.sqrt(len(df_music_metric_values)))
    music1 = df_music_applied.loc['mean'] + df_music_applied.loc['std']
    music2 = df_music_applied.loc['mean'] - df_music_applied.loc['std']

    # Entertainment category mean and standard
    df_entertainment_applied = df_entertainment_metric_values.apply([np.mean, np.std, np.median], axis=0)
    df_entertainment_applied.loc['std'] = df_entertainment_applied.loc['std'].multiply(1/np.sqrt(len(df_entertainment_metric_values)))
    entertainment1 = df_entertainment_applied.loc['mean'] + df_entertainment_applied.loc['std']
    entertainment2 = df_entertainment_applied.loc['mean'] - df_entertainment_applied.loc['std']

    # Plots
    ax.plot(df_music_applied.columns, df_music_applied.loc['mean'], color='blue', label='Music Mean $\mu$')
    ax.fill_between(df_music_applied.columns, y1=music1, y2=music2, alpha=0.5, label='Music $\mu\pm\sigma/\sqrt{N}$')
    if median: ax.plot(df_music_applied.columns, df_music_applied.loc['median'], color='green', label='Music Median')

    ax.plot(df_entertainment_applied.columns, df_entertainment_applied.loc['mean'], color='red', label='Entertainment Mean $\mu$')
    ax.fill_between(df_entertainment_applied.columns, y1=entertainment1, y2=entertainment2, alpha=0.5, label='Entertainment $\mu\pm\sigma/\sqrt{N}$')
    if median: ax.plot(df_music_applied.columns, df_entertainment_applied.loc['median'], color='k', label='Entertainment Median')

    ax.set_title(f'{df_music_applied.columns[-1]} Weeks', fontsize=20)
    ax.set_xlabel('Weeks')
    if first: ax.set_ylabel('$\Delta$Views')
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xticks(ax.get_xticks()[::int(np.ceil(int(df_music_applied.columns[-1])/12))])
    ax.grid(alpha=0.5, linestyle=':')


def calculate_metric_values_all_weeks(df_music, df_entertainment, weeks, metric):
    """
    For each number of weeks specified in "weeks", creates the DataFrame of the metric (using metric_dataframe) for Music and Entertainment categories.
    Then returns a 2-dimensional list where each row contains the Music and Entertainment DataFrames for a given number of weeks.

    Args:
        - df_music: Music time-series DataFrame
        - df_entertainment: Entertainment time-series DataFrame
        - weeks: array of weeks specifying the time frames over which to get metric values
        - metric: quantity of interest for which the DataFrame is created. Default is 'delta_views', but could also be delta_subs for example.
    
    Return: 2D list of DataFrames 
    """

    df_metric_list = []
    for week in weeks:
        df_metric_list.append([metric_dataframe(df_music, week, metric=metric), metric_dataframe(df_entertainment, week, metric=metric)])

    return df_metric_list



def plot_metric_values_all_weeks(df_metric_list, col_number=4, median=False, save_title='Metric Values Subplots'):
    """
    Using plot_metric, plots the evolution of the metric for all number of weeks contained in the list of DataFrames in the input.
    The input is first calculated with calculate_metric_values_all_weeks.

    Args:
        - df_metric_list (list): 2-dimensional list where each row contains the Music and Entertainment DataFrames for a given number of weeks
        - col_number (int): Number of columns of the subplot (default 4)
        - median (boolean): argument of plot_metric that decides if the median is plotted or not (defaulf False)
        - save_title (string): Title to save the figure in figures folder

    Return: None
    """
    # Create Subplots
    rows = int(np.ceil(len(df_metric_list)/col_number))
    fig, axes = plt.subplots(rows, col_number, figsize=(15, 10), sharex=False, sharey=True)

    # Plot all subplots
    for i in range(len(df_metric_list)):
        row = i //4
        col = i % 4

        plot_metric(df_metric_list[i][0], df_metric_list[i][1], ax=axes[row, col], first=(col==0), median=median)
    
    plt.suptitle ('Time Evolution of $\Delta$ Views', fontsize=40, y=1.1)
    plt.legend(loc='upper left', bbox_to_anchor=(-3, -0.25), ncol=2)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)
    data_utils.save_plot(file_name=save_title, overwrite=True)
    plt.show()



def plot_general_mean(df_music, df_ent, metric, log_scale=False):
    """  
    Plots overall mean for a specified metric (typically delta_views or delta_subs) for Music and Entertainment categories on a single figure

    Args:
        - df_music: Music time-series DataFrame
        - df_ent: Entertainment time-series DataFrame
        - metric: String specifying quantity of interest to plot over time
    
    Return: None
    """

    # Music delta_views time-series
    df_music['datetime'] = pd.to_datetime(df_music['datetime'])
    grouped_music = df_music.groupby('datetime')[metric]
    mean_values_music = grouped_music.mean()
    time_points_music = mean_values_music.index

    # Entertainment delta_views time-series
    df_ent['datetime'] = pd.to_datetime(df_ent['datetime'])
    grouped_ent = df_ent.groupby('datetime')[metric]
    mean_values_ent = grouped_ent.mean()
    time_points_ent = mean_values_ent.index

    # Plot
    plt.figure(figsize=(15, 8))
    plt.plot(time_points_music, mean_values_music, label='Music', color='blue', linewidth=2)
    plt.plot(time_points_ent, mean_values_ent, label='Entertainment', color='red', linewidth=2, alpha=0.5)
    plt.title('Time Evolution')
    plt.xlabel('Time')
    plt.ylabel('Mean '+ metric)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=60)
    if log_scale: plt.yscale('log')
    plt.tight_layout()
    plt.show() 



def plot_general_rolling_mean(df_music, df_ent, metric, rolling_window=4, log_scale=False):
    """  
    Plots the rolling mean for a specified metric (typically delta_views or delta_subs) for Music and Entertainment categories on a single figure.

    Args:
        - df_music: Music time-series DataFrame
        - df_ent: Entertainment time-series DataFrame
        - metric: String specifying quantity of interest to plot over time
        - rolling_window: Integer specifying the size of the rolling window (in time points)
        - log_scale: Boolean indicating whether to use a logarithmic scale for the y-axis
    
    Return: None
    """

    # Music delta_views time-series
    df_music['datetime'] = pd.to_datetime(df_music['datetime'])
    grouped_music = df_music.groupby('datetime')[metric]
    rolling_mean_music = grouped_music.mean().rolling(rolling_window, min_periods=1).mean()
    time_points_music = rolling_mean_music.index

    # Entertainment delta_views time-series
    df_ent['datetime'] = pd.to_datetime(df_ent['datetime'])
    grouped_ent = df_ent.groupby('datetime')[metric]
    rolling_mean_ent = grouped_ent.mean().rolling(rolling_window, min_periods=1).mean()
    time_points_ent = rolling_mean_ent.index

    # Plot
    plt.figure(figsize=(15, 8))
    plt.plot(time_points_music, rolling_mean_music, label='Music (Rolling Mean)', color='blue', linewidth=2.5)
    plt.plot(time_points_ent, rolling_mean_ent, label='Entertainment (Rolling Mean)', color='red', linewidth=2.5, alpha=0.7)
    plt.title(f'Time Evolution (Rolling Mean - {rolling_window} Time Points)')
    plt.xlabel('Time')
    plt.ylabel('Mean '+ metric)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=60)
    if log_scale: plt.yscale('log')
    plt.tight_layout()
    plt.show()



def return_to_baseline_analysis(df, metric, baseline_window=4, tolerance=0.1, max_return_time=12, prom_percent=0.2):
    """
    Analyzes the time it takes for a metric (e.g., delta_views or delta_subs) to return to baseline
    after peaks across all channels in a category, stores channel-level results, and aggregates them.

    Args:
        df (pd.DataFrame): Time-series DataFrame for a category, containing multiple channels.
                           Must have 'channel' as one of the columns.
        metric (str): The column name of the metric to analyze (e.g., 'delta_views' or 'delta_subs').
        baseline_window (int): Number of time points to calculate the rolling baseline.
        tolerance (float): Tolerance range around the baseline (e.g., ±10%).
        max_return_time (int): Maximum number of time points to look ahead for return to baseline.
        prom_percent (float): Percentage of baseline that defines the prominence of detected peaks (default 0.2).

    Returns:
        dict: Aggregated results and per-channel outputs for plotting histograms.
    """
    # Group by channel and initialize lists for storing per-channel results
    grouped = df.groupby('channel')
    channel_results = []

    # Get all return times instead of averaging over each channel
    all_return_times = []


    for channel, channel_data in grouped:
        channel_data = channel_data.reset_index(drop=True)  # Ensure clean indexing
        channel_data['baseline'] = channel_data[metric].rolling(window=baseline_window, min_periods=1).mean()
        
        # Detect peaks in the metric
        prominence = prom_percent*np.array(channel_data['baseline'])
        peaks, _ = find_peaks(channel_data[metric], prominence=prominence)
        return_times = []
        num_peaks = len(peaks)

        for peak in peaks:
            peak_value = channel_data[metric].iloc[peak]
            baseline_value = channel_data['baseline'].iloc[peak]
            tolerance_range = (1 - tolerance) * baseline_value, (1 + tolerance) * baseline_value

            # Track how long it takes to return to baseline within tolerance
            for t in range(peak + 1, min(peak + 1 + max_return_time, len(channel_data))):
                # Check if within tolerance range
                if channel_data[metric].iloc[t] <= tolerance_range[1]:
                    # Ensure no other peaks exist between peak and t
                    intervening_peaks = [p for p in peaks if peak < p < t]
                    if not intervening_peaks:
                        return_times.append(t - peak)
                        break
            else:
                # If no return within max_return_time or interrupted by other peaks, record as no return
                return_times.append(np.nan)

        # Compute channel-level metrics
        return_times = np.array(return_times, dtype=float)
        avg_return_time = np.nanmean(return_times) if num_peaks > 0 else np.nan
        std_return_time = np.nanstd(return_times) if num_peaks > 1 else np.nan
        proportion_returning = np.sum(~np.isnan(return_times)) / num_peaks if num_peaks > 0 else np.nan
        
        all_return_times.extend(return_times)

        # Store results for this channel
        channel_results.append({
            'channel': channel,
            'avg_return_time': avg_return_time,
            'std_return_time': std_return_time,
            'proportion_returning': proportion_returning,
            'num_peaks': num_peaks
        })

    # Convert results to a DataFrame for histogram plotting
    results_df = pd.DataFrame(channel_results)

    # Aggregate metrics across channels
    aggregated_results = {
        'avg_return_time': results_df['avg_return_time'].mean(),
        'std_return_time': results_df['avg_return_time'].std(),
        'avg_proportion_returning': results_df['proportion_returning'].mean(),
        'std_proportion_returning': results_df['proportion_returning'].std(),
        'total_peaks': results_df['num_peaks'].sum(),
        'results_df': results_df,  # Include channel-level results for further analysis
        'return_times': np.array(all_return_times)
    }
    return aggregated_results



def return_times_results_and_histograms(music_return_times, entertainment_return_times):
    """
    Prints the Average and Standard Deviation of return times and the average proportion of peaks that return to baseline for
    Music and Entertainment categories. Also plots the histograms of Channel Average Return Times, as well as a histogram of Return Times
    for both Music and Entertainment.

    Args:
        - music_return_times: dictionary containing all information about Music return times
        - entertainment_return_times: dictionary containing all information about Entertainment return times
    
    Return: None
    """

    # Load return times
    return_times_music = music_return_times['return_times']
    return_times_ent = entertainment_return_times['return_times']

    # Filter out Nan values
    return_times_music = return_times_music[~np.isnan(return_times_music)]
    return_times_ent = return_times_ent[~np.isnan(return_times_ent)]

    # Load results dataframe
    results_df_music = music_return_times['results_df']
    results_df_ent = entertainment_return_times['results_df']

    # Access aggregated results for Music
    print(f"Music Mean Channel Average Return Time: {music_return_times['avg_return_time']:.2f} Weeks")
    print(f"Music Error of Mean Return Time: {music_return_times['std_return_time']/len(results_df_music['avg_return_time']):.5f} Weeks")
    print(f"Music Mean Return Time: {np.mean(return_times_music):.2f}")
    print(f"Music Propotion of Return: {music_return_times['avg_proportion_returning']*100:.2f}%")
    print(f'Number of Music Peaks: {len(return_times_music)} \n')

    # Access aggregated results for Entertainment
    print(f"Entertainment Mean Channel Average Return Time: {entertainment_return_times['avg_return_time']:.2f} Weeks")
    print(f"Entertainment Error of Mean Return Time: {entertainment_return_times['std_return_time']/len(results_df_ent['avg_return_time']):.5f} Weeks")
    print(f"Entertainment Mean Return Time: {np.mean(return_times_ent):.2f}")
    print(f"Entertainment Propotion of Return: {entertainment_return_times['avg_proportion_returning']*100:.2f}%")
    print(f'Number of Entertainment Peaks: {len(return_times_ent)}')

    # Plot histogram of per-channel average return times
    plt.hist(results_df_music['avg_return_time'].dropna(), bins=20, alpha=0.7, log=True, label='Music', density=True)
    plt.hist(results_df_ent['avg_return_time'].dropna(), bins=20, alpha=0.4, log=True, label='Entertainment', density=True)
    plt.xlabel('Return Time (weeks)')
    plt.ylabel('Number of Channels')
    plt.title('Histogram of Average Return Times')
    plt.legend()
    plt.show()

    # Plot histogram of return times for Music and Entertainment
    plt.hist(return_times_music, bins=20, alpha=0.7, log=True, label='Music', density=True)
    plt.hist(return_times_ent, bins=20, alpha=0.4, log=True, label='Entertainment', density=True)
    plt.xlabel('Return Time (weeks)')
    plt.ylabel('Number of Entries')
    plt.title('Histogram of Return Times')
    plt.legend()
    plt.show()



def calculate_decay_rates_between_peaks(df, metric, baseline_window=4, tolerance=0.1, prom_percent=0.2):
    """
    Calculates exponential decay rates between consecutive peaks of a specified metric for each channel,
    ensuring the metric returns to a specified baseline before the next peak.

    Args:
        df (pd.DataFrame): Time-series DataFrame for a category containing multiple channels.
                           Must have 'channel' as one of the columns.
        metric (str): The column name of the metric to analyze (e.g., 'delta_views' or 'delta_subs').
        baseline_window (int): Number of time points to calculate the rolling baseline.
        tolerance (float): Tolerance range around the baseline (e.g., ±10%).
        prom_percent (float): Percentage of baseline that defines the prominence of detected peaks (default 0.2).

    Returns:
        dict: Results containing decay rates for each channel and aggregated statistics.
    """

    def exponential_decay(t, y0, k):
        """Exponential decay model for fitting."""
        return y0 * np.exp(-k * t)

    # Group by channel and initialize lists for storing results
    grouped = df.groupby('channel')
    channel_results = []

    # Get all decay rates instead of averaging over channels
    all_decay_rates = []

    for channel, channel_data in grouped:
        channel_data = channel_data.reset_index(drop=True)  # Ensure clean indexing

        # Calculate rolling baseline
        channel_data['baseline'] = channel_data[metric].rolling(window=baseline_window, min_periods=1).mean()

        # Detect peaks in the metric (considering noise threshold)
        prominence = prom_percent*np.array(channel_data['baseline'])
        peaks, _ = find_peaks(channel_data[metric], prominence=prominence)

        # Skip if there are fewer than 2 peaks (no decay can be calculated)
        if len(peaks) < 2:
            continue

        decay_rates = []

        # Iterate over consecutive peaks
        for i in range(len(peaks) - 1):
            peak_start = peaks[i]
            peak_end = peaks[i + 1]

            # Extract baseline value at the current peak
            baseline_value = channel_data['baseline'].iloc[peak_start]
            tolerance_range = (1 - tolerance) * baseline_value, (1 + tolerance) * baseline_value

            # Check for return to baseline before the next peak
            return_to_baseline = False
            for t in range(peak_start + 1, peak_end):
                if channel_data[metric].iloc[t] <= tolerance_range[1]:
                    return_to_baseline = True
                    return_index = t
                    break

            # Skip if no return to baseline was detected
            if not return_to_baseline:
                continue

            # Fit exponential decay from peak_start + 1 to the return_to_baseline index
            fit_window = channel_data.iloc[peak_start + 1 : return_index + 1]

            if len(fit_window) <= 2:  # Avoid fitting if not enough points
                continue

            t = np.arange(len(fit_window))  # Time points (t=0 at the start of the decay)
            y = fit_window[metric].values

            if np.all(y > 0):  # Fit only if all values are positive
                try:
                    # Initial guess: y0 is the first value, k is a small positive number
                    popt, _ = curve_fit(exponential_decay, t, y, p0=[y[0], 0.1], bounds=(0, np.inf))
                    decay_rates.append(popt[1])  # Append the decay rate (k)
                except RuntimeError:
                    continue  # Skip if fitting fails

        # Compute channel-level metrics
        avg_decay_rate = np.mean(decay_rates) if decay_rates else np.nan
        std_decay_rate = np.std(decay_rates) if len(decay_rates) > 1 else np.nan

        # Store results for this channel
        channel_results.append({
            'channel': channel,
            'avg_decay_rate': avg_decay_rate,
            'std_decay_rate': std_decay_rate,
            'num_peaks': len(peaks),
            'num_fitted_peaks': len(decay_rates)
        })

        # Add decay rates to list
        all_decay_rates.extend(decay_rates)

    # Convert results to a DataFrame for further analysis
    results_df = pd.DataFrame(channel_results)

    # Aggregate metrics across channels
    aggregated_results = {
        'avg_decay_rate': results_df['avg_decay_rate'].mean(),
        'std_decay_rate': results_df['avg_decay_rate'].std(),
        'results_df': results_df,  # Include channel-level results for further analysis
        'decay_rates': np.array(all_decay_rates)
    }

    return aggregated_results


def decay_rates_results_and_histograms(music_decay_rates, entertainment_decay_rates):
    """
    Prints the Average and Standard Deviation of Decay Rates for Music and Entertainment categories. 
    Also plots the histograms of Channel Average Decay Rates, as well as a histogram of Decay Rates for both Music and Entertainment.

    Args:
        - music_return_times: dictionary containing all information about Music decay rates
        - entertainment_return_times: dictionary containing all information about Entertainment decay rates
    
    Return: None
    """

    # Load decay rates
    ent_decay_peaks_all = entertainment_decay_rates['decay_rates']
    music_decay_peaks_all = music_decay_rates['decay_rates']

    # Load results dataframes
    music_decay_peaks_results = music_decay_rates['results_df']
    ent_decay_peaks_results = entertainment_decay_rates['results_df']

    # Access aggregated results
    print(f"Music Mean Channel Average Decay Rate: {music_decay_rates['avg_decay_rate']:.3f} 1/Weeks")
    print(f"Music Error of Decay Rate: {music_decay_rates['std_decay_rate']/len(music_decay_peaks_results['avg_decay_rate']):.6f} 1/Weeks")
    print(f"Music Mean Decay Rate: {np.mean(music_decay_peaks_all):.3f} 1/Weeks")
    print(f"Number of Peaks for Music: {music_decay_peaks_results['num_fitted_peaks'].sum()} \n")

    # Access aggregated results
    print(f"Entertainment Mean Channel Average Decay Rate: {entertainment_decay_rates['avg_decay_rate']:.3f} 1/Weeks")
    print(f"Entertainment Error of Decay Rate: {entertainment_decay_rates['std_decay_rate']/len(music_decay_peaks_results['avg_decay_rate']):.6f} 1/Weeks")
    print(f"Entertainment Mean Decay Rate: {np.mean(ent_decay_peaks_all):.3f} 1/Weeks")
    print(f"Number of Peaks for Entertainment: {ent_decay_peaks_results['num_fitted_peaks'].sum()}")

    # Histogram of per-channel average decay rates    
    plt.hist(music_decay_peaks_results['avg_decay_rate'].dropna(), bins=20, alpha=0.7, log=True, label='Music', density=True)
    plt.hist(ent_decay_peaks_results['avg_decay_rate'].dropna(), bins=20, alpha=0.4, log=True, label='Entertainment', density=True)
    plt.xlabel('Decay Rate (1/Weeks)')
    plt.ylabel('Number of Channels')
    plt.title('Histogram of Average Decay Rates')
    plt.legend()
    plt.show()

    # Histogram of decay rates for Music and Entertainment
    plt.hist(music_decay_peaks_all, bins=20, alpha=0.7, log=True, label='Music', density=True)
    plt.hist(ent_decay_peaks_all, bins=20, alpha=0.4, log=True, label='Entertainment', density=True)
    plt.xlabel('Decay Rate (1/Weeks)')
    plt.ylabel('Number of Entries')
    plt.title('Histogram of Decay Rates')
    plt.legend()
    plt.show()



def calculate_peak_heights_baseline(df, metric, baseline_window=4, prom_percent=0.2):
    """
    Calculates the height of each peak above a baseline for all channels in the given category dataset.

    Args:
        df (pd.DataFrame): Time-series DataFrame containing data for all channels in a category.
                           Must have 'channel' as one of the columns.
        metric (str): The column name of the metric to analyze (e.g., 'delta_views' or 'delta_subs').
        baseline_window (int): Number of time points before each peak to calculate the baseline.
        prom_percent (float): Percentage of baseline that defines the prominence of detected peaks (default 0.2).

    Returns:
        dict: A dictionary containing:
            - 'peak_heights': NumPy array of all peak heights above baseline across all channels.
            - 'results_df': A DataFrame with per-channel peak height details.
    """

    # Group by channel and initialize lists for storing results
    grouped = df.groupby('channel')
    channel_results = []

    # Collect all peak heights across channels
    all_peak_heights = []

    for channel, channel_data in grouped:
        channel_data = channel_data.reset_index(drop=True)  # Ensure clean indexing

        # Detect peaks in the metric (considering noise threshold)
        baseline = channel_data[metric].rolling(window=baseline_window, min_periods=1).mean()
        prominence = prom_percent*np.array(baseline)
        peaks, properties = find_peaks(channel_data[metric], prominence=prominence)

        # Skip if no peaks are found
        if len(peaks) == 0:
            continue

        # Calculate baseline and peak heights
        peak_heights = []
        for peak_idx in peaks:
            if peak_idx >= baseline_window:  # Ensure sufficient data for baseline calculation
                baseline = channel_data[metric][peak_idx - baseline_window:peak_idx].mean()
                peak_height = channel_data[metric].iloc[peak_idx] - baseline
                if peak_height > 0.001: peak_heights.append(peak_height)

        # Skip channels with no valid peaks
        if len(peak_heights) == 0:
            continue

        # Store per-channel results
        channel_results.append({
            'channel': channel,
            'avg_peak_height': np.mean(peak_heights),
            'std_peak_height': np.std(peak_heights),
            'num_peaks': len(peak_heights)
        })

        # Add to global peak heights
        all_peak_heights.extend(peak_heights)

    # Convert results to a DataFrame for further analysis
    results_df = pd.DataFrame(channel_results)

    # Aggregate metrics across channels
    aggregated_results = {
        'peak_heights': np.array(all_peak_heights),
        'results_df': results_df
    }

    return aggregated_results


def peak_heights_histogram(music_peaks, entertainment_peaks):
    """
    Plots the histogram of Peak Heights above the rolling baseline for Music and Entertainment

    Args:
        - music_peaks: Array containing all peak heights for Music
        - entertainment_peaks: Array containing all peak heights for Entertainment
    """

    # Get bins of normal-scale histogram
    hist_music, bins_music, _ = plt.hist(music_peaks['peak_heights'], bins=20, density=False)
    hist_ent, bins_ent, _ = plt.hist(entertainment_peaks['peak_heights'], bins=20, density=False)
    plt.close()

    # Histogram on log scale. 
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins_music = np.logspace(np.log10(bins_music[0]),np.log10(bins_music[-1]),len(bins_music))
    logbins_ent = np.logspace(np.log10(bins_ent[0]),np.log10(bins_ent[-1]),len(bins_ent))

    print(f"Number of Peaks for Music: {len(music_peaks['peak_heights'])}")
    print(f"Number of Peaks for Entertainment: {len(entertainment_peaks['peak_heights'])}")

    fig = plt.figure(figsize=(15, 8))
    plt.hist(entertainment_peaks['peak_heights'], bins=logbins_ent, alpha=0.7, log=True, label='Entertainment', density=False, color='darkorange')
    plt.hist(music_peaks['peak_heights'], bins=logbins_music, alpha=0.7, log=True, label='Music', density=False, color='dodgerblue')
    plt.xlabel('Peak Height')
    plt.ylabel('Number of Peaks')
    plt.title('Histogram of Peak Heights Above Baseline - Log Scale')
    plt.xscale('log')
    plt.legend()
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


def plot_time_series_one_channel(df_time_series_one_channel, metric, prom_percent, channel_name, rolling_window, tolerance=0.1, max_return_time=12):
    
    # Put datetime in correct form
    df_time_series_one_channel['datetime'] = pd.to_datetime(df_time_series_one_channel['datetime'])
    df_time_series_one_channel = df_time_series_one_channel.sort_values('datetime')
    
    # Find peaks
    df_time_series_one_channel['baseline'] = df_time_series_one_channel[metric].rolling(rolling_window, min_periods=1, center=False).mean()
    prominence = prom_percent*np.array(df_time_series_one_channel['baseline'])
    peaks, _ = find_peaks(df_time_series_one_channel[metric], prominence=prominence)

    # Peak heights and time
    peak_times = df_time_series_one_channel['datetime'].iloc[peaks]
    peak_heights = df_time_series_one_channel[metric].iloc[peaks]

    # Video upload times
    video_times = df_time_series_one_channel.loc[df_time_series_one_channel['delta_videos']>=1, 'datetime']
    video_height = np.full(len(video_times), 0)

    # Return times
    return_times = []
    for peak in peaks:
        peak_value = df_time_series_one_channel[metric].iloc[peak]
        baseline_value = df_time_series_one_channel['baseline'].iloc[peak]
        tolerance_range = (1 - tolerance) * baseline_value, (1 + tolerance) * baseline_value

        # Track how long it takes to return to baseline within tolerance
        for t in range(peak + 1, min(peak + 1 + max_return_time, len(df_time_series_one_channel))):
            if df_time_series_one_channel[metric].iloc[t] <= tolerance_range[1]:
                return_times.append(t)
                break
        else:
            # If no return within max_return_time, record as no return
            return_times.append(np.nan)

    return_times = np.array(return_times, dtype=float)
    return_times = return_times[~np.isnan(return_times)]
    return_datetime = df_time_series_one_channel['datetime'].iloc[return_times]
    return_height = df_time_series_one_channel[metric].iloc[return_times]

    # Plot delta_views evolution
    plt.figure(figsize=(15, 8))
    plt.plot(df_time_series_one_channel['datetime'], df_time_series_one_channel[metric], label='Music', color='blue', linewidth=2)
    plt.title(f'Time Evolution of {metric} for {channel_name}')
    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=60)

    # Plot peaks, rolling average, video times and return times
    plt.scatter(peak_times, peak_heights, label='Peaks', color='red', marker='x', s=2, linewidth=20)
    plt.plot(df_time_series_one_channel['datetime'], df_time_series_one_channel['baseline'], 
             label='Rolling Average', linestyle = '--', color='forestgreen', linewidth='2')
    plt.plot(return_datetime, return_height, color='orange', marker='o', markersize='10', linestyle='', label='Return Time', alpha=0.75)
    plt.scatter(video_times, video_height, color='k', marker='+', linewidth=12, s=2, label='Published Video')
    plt.legend()
    plt.tight_layout()
    plt.show()



def filter_and_categorize(df_channels, df_time_series, p_threshold=0.9, top=True):
    """
    Filters the time-series DataFrame to only include the channels that have together amass p_threshold of the total number of subscribers
    (we only consider the Music and Entertainment categories)

    Args:
        - df_channels (pd.DataFrame): Channels DataFrame
        - df_time_series (pd.DataFrame): Time-series DataFrame
        - p_threshold (float): Fraction of subscribers to consider for top/bottom channels
        - top (boolean): If True takes top fraction of channels, if False takes bottom fraction

    Return:
        - df_filtered_music (pd.DataFrame): Filtered Music time-series DataFrame
        - df_filtered_entertainment (pd.DataFrame): Filtered Entertainment time-series DataFrame
    """
    # Get only Music and Entertainment categories
    df_channels_music_ent = df_channels.loc[df_channels['category_cc'].isin(['Music', 'Entertainment'])]

    # Filter channels by proportion
    df_channels_filtered = filter_channels_by_proportion(df_channels_music_ent, p_threshold=p_threshold, top=top)

    # Filter time-series by top/bottom channels
    df_time_series_filtered = df_time_series.loc[df_time_series['channel'].isin(df_channels_filtered['channel'])]

    # Filtered Music time-series DataFrame
    df_filtered_music = df_time_series_filtered.loc[df_time_series_filtered['category']=='Music']

    # Filtered Entertainment time-series DataFrame
    df_filtered_entertainment = df_time_series_filtered.loc[df_time_series_filtered['category']=='Entertainment']

    return df_filtered_music, df_filtered_entertainment



def plot_channel_time_series(df_channels, df_time_series, channel_name, metric, prom_percent=0.2, rolling_window=4, tolerance=0.1, max_return_time=10):
    """
    Finds and plots the time-series of a specified channel for a given metric. Also plots the rolling average, detected peaks, return times
    and video upload dates.

    Args:
        - df_channels (pd.DataFrame): Channels DataFrame
        - df_time_series (pd.DataFrame): Time-series DataFrame
        - channel_name (string): Name of the channel of interest
        - metric (string): Quantity of time-series DataFrame to plot
        - prom_percent (float): Percentage of baseline that defines the prominence of detected peaks (default 0.2).
        - rolling_window (int): Number of time points to calculate the rolling baseline.
        - tolerance (float): Percentage of baseline value where a peak is detected as "returned to baseline"
        - max_return_time (int): Maximum allowed number of weeks to look for a peak returning to baseline

    Return: None
    """
    # Get channel ID
    channel_id = df_channels.loc[df_channels['name_cc']==channel_name, 'channel'].iloc[0]

    # Get channel time-series
    channel_time_series = df_time_series.loc[df_time_series['channel']==channel_id]

    # Plot the time-series
    plot_time_series_one_channel(channel_time_series, 
                             metric=metric, 
                             prom_percent=prom_percent,  
                             channel_name=channel_name, 
                             rolling_window=rolling_window, 
                             tolerance=tolerance, 
                             max_return_time=max_return_time)
    


