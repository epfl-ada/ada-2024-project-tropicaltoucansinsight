# The YouTube Heavyweights: Entertainment vs. Music Face Off

## Abstract

Some claim the greatest 21st-century showdown was Floyd Mayweather versus Logan Paul, packed with stakes in marketing, money, and public hype. We couldn’t disagree more: the real battle is Entertainment vs. Music on YouTube! Leveraging the YouNiverse dataset, a massive collection of metadata covering 136k channels, 72.9M videos, and 2.8 years of time series data on views and subscribers, we dive into YouTube’s top two categories, analyzing their reach through views, subscriber counts, and strategic collaborations. Do entertainment creators ramp up content in December to maximize ad revenue? Do music artists dominate the long game thanks to loyal fan bases? From seasonal trends to community dynamics, we’ll explore how these giants shape and reshape their audiences. Get ready for a data showdown where each side fights for the throne of influence, popularity, and engagement. Through time series analysis, hypothetical monetization, and network insights, this is YouTube’s ultimate battle—where only one category can claim the crown in the world’s biggest digital arena!

## Research Questions and Methods
With this project, we aim to answer the following research questions (rephrased to identify a clear winner for each question). After each question, we detail the different methods we plan to use:
1. **Which category captures greater monetization potential, and does either Entertainment or Music benefit more from ad-friendly periods like December, while suffering less during slower periods like summer?**
   - After gathering information from [Google Support](https://support.google.com/youtube?sjid=13108256786547781650-EU#topic=9257498), we quantify monetization potential ($\text{MP}$) by creating a metric defined as

     $$\text{MP}(\text{video})= c_{\rm cat}\cdot \text{round}\left(\frac{N_{\text{views}}}{1000}\right)\cdot\left[ 1+\alpha\cdot \text{round}\left(  \frac{t_{\text{video}} \text{ [min]}}{8\text{ [min]}}  \right)  \right],$$ 

     where $c_{\rm cat}$ represents the earnings associated with a specific category per thousand views, $N_{\text{views}}$ is the number of views of the video, $t_{\text{video}}$ is the video duration in minutes, $\alpha$ is a scaling factor (set to 0.5 for now), and $\text{round}(\cdot)$ rounds its argument to the nearest integer. The larger the value of $\alpha$, the more influence video duration has on monetization potential. We will compare this metric across the full dataset and during key periods, hypothesizing that Entertainment may experience higher traffic peaks during holiday seasons, while Music might maintain steady viewership through consistent fan support.
2. **Which category, Entertainment or Music, offers a broader diversity of content types and formats, and does this diversity lead to a measurable advantage in audience engagement and retention?**
   - We will categorize video formats and themes within each category using a machine learning approach, such as keyword clustering with tools like SpaCy or RoBERTa. If feasible, we will also incorporate video duration as a feature for clustering.  We then evaluate how diversity correlates with engagement metrics to determine if one category benefits from a richer spectrum of content types.
   - Furthermore, we will use an LLM to analyze the sentiment (positivity/negativity) of the content in both categories. This can be either done overall or for the different formats and themes established before. It will be interesting to relate sentiment to engagement.
3. **Do collaboration patterns in Music or Entertainment channels yield greater viewership, and which category leverages collaborations more effectively to expand reach?**
   - Using text mining on video titles, descriptions and tags, we can identify collaborations (keywords like ‘feat’, ‘ft’, ‘with’, ‘w/’) and measure their impact on viewership and reach. We can then compare the dominance and success of collaborations within each category to assess which leverages partnerships in the most effective way.
4. **Which category, Music or Entertainment, maintains more consistent popularity over both short-term and long-term timeframes, and which one performs better in sustaining viewer interest?**
   - We measure short-term versus long-term popularity by setting different time frames (e.g. 24h, 1 week, 1 month, 6 month, 1 year) on the time-series data. By analyzing view trends over time, we identify whether Music or Entertainment videos sustain interest longer, and we classify outliers, like channels with high views but low subscribers, to determine the loyalty differences.
5. **Which category, Entertainment or Music, demonstrates more effective seasonal release patterns that lead to higher viewership peaks?**
   - Using the time-series data, we analyze dynamics through time to determine if one category, Entertainment or Music, aligns its content releases with periods of high potential viewership like December. Here, it will be interesting to provide examples of some of the bigger channels in each category to give a concrete visualization of these trends.
6. **In which category does major content release (e.g., albums for Music or viral projects for Entertainment) lead to stronger subscriber growth and higher engagement?**
   - Focussing on significant releases, we track subscriber jumps around events like album drops for Music or viral videos for Entertainment. Time-series analysis of weekly subscriber data helps measure which category sees stronger engagement and audience growth in response to these major releases. 
7. **Optional: What are the differences between communities in between Music and Entertainment videos and do they overlap? What are their different characteristics, such as size and duration in time? What can we say with the answers to the other questions above?**
    - If our computers allow, with the use of Pyspark we will try to find out which communities populate the Music and Entertainment categories (also if there is some crossover) and try to figure out how they evolve in time. With all the information from the previous points (such as engagement and content diversity), we can understand how communities in both categories behave. 

### Remarks about Data Preparation and Cleaning
- We begin by filtering the YouNiverse dataset to include only channels and videos within Entertainment and Music categories.
- Essential metadata such as view counts, publications dates, and content/collaboration indicators (e.g. title, tags and descriptions) among others are extracted. We drop irrelevant data (for instance crawl date) to facilitate processing given the dataset’s large disk space requirements.
- Time-series data are organized in the same manner to capture trends across the  dataset’s 2.8-year span, useful to capture seasonal patterns and viewership fluctuations. In addition, basic statistics about both categories will be given, such as time-series of total view and subscriber counts.

## Proposed timeline and Organization within the Team
```mermaid
gantt
    title Project Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d, %Y

    section Data Handling
    Data handling, preprocessing, exploratory data analysis :a1, 2024-11-01, 20d

    section Website Setup
    Set up the website :a2, 2024-11-16, 5d

    section Implementation
    Tasks implementation and preliminary analysis :a3, 2024-11-21, 15d

    section Analysis
    Compile final analysis :a4, 2024-12-04, 7d

    section Redaction
    Data story redaction :a5, 2024-12-06, 14d

    section Submission
    Project final submission :a6, 2024-12-18, 2d
```

| **Task**| **Research Question**| **Description**| **Assigned Member** |
|:--:|:--:|--|:--:|
| **Define Monetization Metric (MP)**| 1 | Develop and calculate the monetization potential metric | Timothée |
| **Quantify Monetization During Key Periods**| 1 | Analyze MP across the entire dataset and for specific periods like December and summer, testing the holiday impact on Entertainment and Music categories| Timothée |
| **Categorize Video Types and Formats**| 2 | Use keyword clustering to categorize formats within each category using tools like SpaCy or RoBERTa | Jérémy |
| **Evaluate Engagement and Diversity Correlation**| 2 | Assess how diversity in content types affects engagement metrics for both categories | Jérémy |
| **Sentiment Analysis for Content Types**| 2| Analyze the sentiment of videos across themes, measuring positivity/negativity and its impact on engagement (define engagement) | Sylvain |
| **Identify and Analyze Collaboration Patterns**| 3 | Extract keywords in titles/descriptions to identify collaborations | Sylvain |
| **Short-term vs Long-term Popularity Trends** | 4 | Analyze time-series data to assess short-term vs. long-term interest. Define a loyalty metric and examine it | Max |
| **Seasonal Viewership Pattern Analysis** | 5 | Examine time-series data for seasonal trends in content releases, identifying peaks in periods like December for each category | Max |
| **Case Study of Major Channels** | 5 | Select major channels in each category to provide examples that illustrate seasonal trends and highlight key patterns | Everyone |
| **Track Subscriber Growth for Major Releases** | 6 | Measure subscriber increases around significant releases | Everyone |
| **Community Analysis Using Pyspark (Optional)**| 7 (Optional) | Use Pyspark to explore community structures | Everyone |

---

# Quickstart

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

Clone the project repository and navigate into the project directory:

```bash
git clone git@github.com:epfl-ada/ada-2024-project-tropicaltoucansinsight.git
cd ada-2024-project-tropicaltoucansinsight
```

### 2. Set Up the Conda Environment

Install the environment depending on your operating system:

- **For Windows**:
  ```bash
  conda env create -f environments/windows_env.yml
  ```

- **For macOS**:
  ```bash
  conda env create -f environments/mac_env.yml
  ```

### 3. Activate the Environment

Once the environment is created, activate it:

```bash
conda activate tropicaltoucansinsight
```

You’re all set!


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── environments                <- Environment files
│
├── experiments                 <- Notebooks of tests
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── results.ipynb               <- Notebook of results
│
├── .gitignore                  <- List of files ignored by git
└── README.md
```
