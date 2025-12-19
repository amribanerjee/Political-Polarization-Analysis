
```markdown
# Political Speech Analysis Toolkit

A collection of Python-based Natural Language Processing (NLP) tools designed for the comparative and longitudinal analysis of political rhetoric. These tools are optimized for use in **Jupyter Notebooks** or **Google Colab**.

## 🛠 Features

### 1. Comparative Analysis (`analysis.py`)
Designed for head-to-head comparison between two specific speeches (e.g., Democrat vs. Republican nominees).
* **Sentiment Analysis**: Utilizes the VADER Lexicon to provide positivity, neutrality, negativity, and compound scores.
* **Vocabulary Overlap**: Calculates **Jaccard Similarity** to measure unique word overlap.
* **Semantic Divergence**: Uses **TF-IDF Vectorization** and **Cosine Similarity** to determine statistical similarity in word usage patterns.

### 2. Corpus Frequency Analyzer (`frequency-analyzer.py`)
Built for longitudinal studies across multiple election cycles from 1944 to 2024.
* **Bulk Processing**: Provides an interactive grid for uploading Republican and Democratic speeches by year.
* **Lemma-Based Counting**: Uses `spaCy` to reduce words to their root forms (lemmatization) for accurate frequency tracking.
* **Corpus Insights**: Identifies the most dominant themes and provides a Top 10 frequency breakdown across the entire dataset.

---

## 🚀 Installation

Ensure you have Python installed, then run the following commands to set up the required dependencies:

```bash
pip install spacy nltk numpy ipywidgets
python -m spacy download en_core_web_md

```

---

## 📋 Usage Instructions

### Using the Comparative Analyzer (`analysis.py`)

1. Run the script in a Jupyter cell to initialize the `PoliticalSpeechAnalyzer` class.
2. Use the **Upload** buttons to provide two `.docx` files.
3. The script will automatically process the text, remove "stop words" and punctuation, and display a detailed similarity and sentiment report.

### Using the Frequency Analyzer (`frequency-analyzer.py`)

1. Run the script in a Jupyter cell to generate the year-based upload grid.
2. Upload any number of speeches into the provided year/party slots (1944–2024).
3. Click **"ANALYZE CORPUS"** to generate a frequency breakdown of the most common words and their percentage of use per document.

---

## ⚙️ Technical Details

* **Language Model**: Both scripts utilize `spaCy`'s `en_core_web_md` model for linguistic processing.
* **Text Extraction**: Uses `xml.etree.cElementTree` to parse raw text directly from the underlying XML structure of `.docx` files.
* **Preprocessing**: Includes lowercasing, lemmatization, and the removal of non-alphabetic characters to ensure clean data analysis.

---

## ⚠️ Requirements

* **Environment**: Jupyter Notebook, JupyterLab, or VS Code with Jupyter extension.
* **File Format**: Only `.docx` files are supported.

```

```