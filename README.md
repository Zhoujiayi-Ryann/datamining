# datamining

## Overview

This project involves sentiment analysis using various techniques and models. The main scripts included are `sentiment.py` and `sentiment3.py`, which demonstrate different approaches to sentiment analysis on the IMDB dataset.

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.x
- Required Python packages: pandas, nltk, scikit-learn, matplotlib, seaborn, wordcloud

You can install the necessary packages using pip:

```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud
```

Additionally, ensure that the NLTK data required for sentiment analysis is downloaded. This includes the VADER lexicon, stopwords, punkt tokenizer, and WordNet.

## Running the Scripts

### sentiment.py

This script performs sentiment analysis using an ensemble model that combines VADER sentiment scores, SentiML features, and TF-IDF features. It also includes visualization of the results.

To run the script:

1. Ensure the IMDB dataset is available as `IMDBDataset.csv` in the same directory.
2. Execute the script using Python:

```bash
python sentiment.py
```

The script will output the results of the sentiment analysis, including accuracy scores and visualizations saved in the `visualization_results` directory.

### sentiment3.py

This script demonstrates sentiment analysis using VADER and a custom SentiML framework, as well as a traditional machine learning approach with TF-IDF and logistic regression.

To run the script:

1. Ensure the IMDB dataset is available as `IMDBDataset.csv` in the same directory.
2. Execute the script using Python:

```bash
python sentiment3.py
```

The script will output the results of the sentiment analysis, including accuracy scores and visualizations saved in the `visualization_results` directory.

## Visualization

Both scripts generate various visualizations, including confusion matrices, feature importance plots, and word clouds. These are saved in the `visualization_results` directory for further analysis.

## Notes

- Ensure that the NLTK data is downloaded before running the scripts. The scripts include a function to check and download the necessary data if not already present.
- The scripts assume the IMDB dataset is formatted correctly and available in the specified location.