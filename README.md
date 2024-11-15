# The YouTube Heavyweights: Entertainment vs. Music Face Off

## Abstract

Some say the greatest showdown of the 21st century was Floyd Mayweather versus Logan Paul, with massive stakes in marketing, money, and public hype. We couldn’t disagree more: the real battle is Entertainment vs. Music on YouTube! Leveraging the YouNiverse dataset, a massive collection of metadata covering 136k channels, 72.9M videos, and 2.8 years of time series data on views and subscribers, we dive into YouTube’s top two categories, analyzing their reach through views, subscriber counts, and strategic collaborations. Do entertainment creators ramp up content in December to maximize ad revenue? Do music artists dominate the long game thanks to loyal fan bases? From seasonal trends to community dynamics, we’ll explore how these giants shape and reshape their audiences. Get ready for a data showdown where each side fights for the throne of influence, popularity, and engagement. Through time series analysis, hypothetical monetization, and network insights, this is YouTube’s ultimate battle—where only one category can claim the crown in the world’s biggest digital arena!

## Research Questions and Methods
With this project, we would like to answer the following questions (note: rephrased in a way to have a clear winner for each research question in the end). After each question, we detail the different methods we plan to use:
1. **Which category captures greater monetization potential, and does either Entertainment or Music benefit more from ad-friendly periods like December, while suffering less during slower periods like summer?**
   - After gathering some [info online](https://support.google.com/youtube?sjid=13108256786547781650-EU#topic=9257498), we quantify monetization potential ($\text{MP}$) by creating a metric defined as

     $$\text{MP}(\text{video})= N_{\text{views}}\cdot\left[ 1+\alpha\cdot \text{round}\left(  \frac{t_{\text{video}} \text{ [min]}}{8\text{ [min]}}  \right)  \right],$$ 

     where $N_{\text{views}}$ represents the number of views of the video,  $t_{\text{video}}$ is the video duration, $\alpha$ is a scaling factor (set to 0.5 for now), and $\text{round}(\cdot)$ rounds its argument to the nearest integer. The larger the value of $\alpha$, the more influence video duration has on the monetization potential. Comparisons are made over the full dataset, and across key periods, hypothesizing that Entertainment might see higher traffic peaks during holiday seasons, while Music might retain steady viewership through continuous fan support.
2. **Which category, Entertainment or Music, offers a broader diversity of content types and formats, and does this diversity lead to a measurable advantage in audience engagement and retention?**
   - To assess content diversity, we categorize video formats and themes within each category. To do so, we will try to implement a machine learning approach by keyword clustering, with the use of Spacy or RoBERTa, for example. If possible, we will also try to take into account video duration as a characteristic for clustering.  We then evaluate how diversity correlates with engagement, and whether it plays a role in maintaining the audience’s interest, testing if one category benefits from a richer spectrum of content types.
   - Furthermore, we will use an LLM to analyze the positivity/negativity of the content in both categories. This can be done overall and for the different sub-categorization established before. It will be interesting to relate sentiment to engagement. (TODO: rephrase)
3. **Do collaboration patterns in Music or Entertainment channels yield greater viewership, and which category leverages collaborations more effectively to expand reach?**
   - Using text mining on video titles, descriptions and tags, we identify collaborations (keywords like ‘feat’, ‘ft’, ‘with’, ‘w/’) and measure their impact on viewership and reach. We compare the dominance and success of collaborations within each category to assess which leverages partnerships in the most effective way.
4. **Which category, Music or Entertainment, maintains more consistent popularity over both short-term and long-term timeframes, and which one performs better in sustaining viewer interest?**
   - We measure short-term versus long-term popularity by setting different time frames (e.g. 24h, 1 week, 1 month, 6 month, 1 year) on the time-series data. By analyzing view trends over time, we identify whether Music or Entertainment videos sustain interest longer, and we classify outliers, like channels with high views but low subscribers, to determine the loyalty differences.
5. **Which category, Entertainment or Music, demonstrates more effective seasonal release patterns that lead to higher viewership peaks?**
   - Using the time-series data, we analyze dynamics through time to determine if one category, Entertainment or Music, aligns its content releases with periods of high potential viewership like December. Here, it will be interesting to provide examples of some of the bigger channels in each category to give a concrete visualization of these trends.
6. **In which category does major content release (e.g., albums for Music or viral projects for Entertainment) lead to stronger subscriber growth and higher engagement?**
   - Focussing on significant releases, we track subscriber jumps around events like album drops for Music or viral videos for Entertainment. Time-series analysis of weekly subscriber data helps measure which category sees stronger engagement and audience growth in response to these major releases. (préciser s’il s’agit d’une analyse globale ou sur certains artistes en particulier? par exemple avec un sampling sur des chaînes?)
7. **Optional: What are the differences between communities in between Music and Entertainment videos and do they overlap? What are their different characteristics, such as size and duration in time? What can we say with the answers to the other questions above?**
    - If our computers allow, with the use of Pyspark we will try to find out which communities populate the Music and Entertainment categories (also if there is some crossover) and try to figure out how they evolve in time. With all the information from the previous points (such as engagement and content diversity), we can understand how communities in both categories behave. (TODO: reformulate and find more specific ideas)

### Remarks about Data Preparation and Cleaning
- We begin by filtering the YouNiverse dataset to include only channels and videos within Entertainment and Music categories.
- Essential metadata such as view counts, publications dates, and content/collaboration indicators (e.g. title, tags and descriptions) among others are extracted. We drop irrelevant data (for instance crawl date, comments (TODO: à voir si on garde?)) to facilitate processing given the dataset’s large disk space requirements.
- Time-series data are organized in the same manner to capture trends across the  dataset’s 2.8-year span, useful to capture seasonal patterns and viewership fluctuations. In addition, basic statistics about both categories will be given, such as time-series of total view and subscriber counts.


## Proposed timeline
## Organization
## Questions for TAs (optional): Add here any questions you have for us related to the proposed project.


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
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```
