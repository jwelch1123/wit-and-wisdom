# Wit, Wisdom, and Vector Embeddings

![Static Badge](https://img.shields.io/badge/Python-grey?logo=python) 
![Static Badge](https://img.shields.io/badge/Dash-grey?logo=plotly) 
![Static Badge](https://img.shields.io/badge/Plotly_Express-grey?logo=plotly)
![Static Badge](https://img.shields.io/badge/scikit--learn-grey?logo=scikitlearn)
![Static Badge](https://img.shields.io/badge/SentenceTransformers-grey)



## Description
"Wit, Wisdom, and Vector Embeddings" explores the timeless wisdom contained within Ben Franklin's "Poor Richard's Almanack," which from 1732 to 1758 offered a mix of farming advice, calendars, poems, and proverbs on human nature and morality. This project uses modern data science techniques to cluster these sayings by their meaning; vector embeddings and dimensionality reduction are employed to visually map the similarities among Franklin's insights.

Inspired by the thematic grouping of *Wit & Wisdom from Poor Richard's Almanack* curated by Dover Publications. Data was scrapped from [Wikiquote of Poor Richard's Almanak](https://en.m.wikiquote.org/wiki/Poor_Richard%27s_Almanack) and guided using resources like [ansonyuu's project](https://github.com/ansonyuu/matchmaking) and [StatQuest's UMAP guide](https://youtu.be/eN0wFzBA4Sc), this endeavor presents an interactive visualization where each proverb is plotted according to its thematic essence. Users can engage with Franklin's wisdom, exploring how these historical insights cluster together and even finding the proverb closest to their own input.

This project offers a unique lens to appreciate and explore these proverbs. However many of the phrases and their meanings are "products of their time" and should be understood as interesting sayings rather than direct advice for the modern day.

## Installation

To get started with this repo, you'll need to install several dependencies. This project is built using Python and requires the following libraries:

- Dash
- Plotly
- Pandas
- Scipy
- Sentence Transformers
- UMAP (package name is `umap-learn`)
- Scikit-learn (sklearn)

You can install these dependencies using pip:

```bash
pip install dash plotly pandas scipy sentence-transformers umap-learn scikit-learn
```

## Usage

### Data 
The collection of quotes was sourced from [Wikiquote of Poor Richard's Almanak](https://en.m.wikiquote.org/wiki/Poor_Richard%27s_Almanack) due to a lack of easily available scanned versions of the almanacks. The publication year and text of the quote were collected. New lines were removed and quotes were cleaned of transcription errors and special characters. These quotes are stored in the `wit_and_wisdomes.csv` file with the column names Source Link, Date, and Quote.

### Viewing the Static Version
A Jupyter Notebook (`wit-n-wisdoms.ipynb`) is included in the project files, allowing you to view a static version of the analysis and visualizations. You can open this notebook in Jupyter Lab, Jupyter Notebook, or VSCode (with the proper extensions) to explore the project's concepts and outcomes.

### Running the Interactive Website
To experience the interactive website created with Dash and Plotly, run the `app.py` file. This will start a local server and serve the interactive application, allowing you to explore the vector embeddings in a dynamic and visual format.

```bash
python app.py
```

Once the server is running, navigate to the URL provided in your terminal (typically `http://127.0.0.1:8050/`) to view the application.

## Features
- **Vector Embeddings with LLMs**: Utilize state-of-the-art language models to generate vector embeddings of text data, capturing the semantic nuances. Currently, the *all-MiniLM-L6-v2* from Sentence-Transformers is used but other techniques or LLMs can be swapped in.
- **Dimensionality Reduction with UMAP**: Apply UMAP to reduce the dimensionality of embeddings, making it possible to visualize complex relationships in lower-dimensional space.
- **Interactive Visualizations**: Explore interactive visualizations powered by Dash and Plotly, offering insights into the structure and clusters within the text data.

## Project Status
This project is currently in a functional state, ready for exploration and use. Future improvements will include alternative sentence transformers, alternative dimension reduction, and a better font.

## Credits
Photo by [Vinicius "amnx" Amano](https://unsplash.com/@viniciusamano?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/yellow-and-white-christmas-tree-6ZwDvEabfmE?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash)
  
