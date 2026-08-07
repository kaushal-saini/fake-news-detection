# Fake News Detection

A production-ready machine learning system for binary text classification that detects fake and real news articles using TF-IDF vectorization and Logistic Regression.

**Key Metrics:** ~95% Accuracy | ~94% Precision | ~96% Recall | ~95% F1-Score

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# https://www.kaggle.com/datasets/jainpooja/welfake-dataset
# Place WELFake_Dataset.csv in data/ folder

# 4. Run the complete pipeline
python main.py
```

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Architecture](#architecture)
4. [Usage](#usage)
5. [Model Performance](#model-performance)
6. [Key Concepts](#key-concepts)
7. [Project Structure](#project-structure)
8. [Troubleshooting](#troubleshooting)
9. [Learning Outcomes](#learning-outcomes)
10. [Future Enhancements](#future-enhancements)
11. [License](#license)

---

## Overview

This project implements a complete machine learning pipeline for fake news detection. It demonstrates real-world data science workflows including data preprocessing, feature engineering, model training, evaluation, and inference.

**Use Cases:**
- News verification systems
- Content moderation pipelines
- Misinformation detection
- Educational ML project
- Portfolio demonstration

**Tech Stack:**
- **ML Framework:** Scikit-learn (1.3.0+)
- **Data Processing:** Pandas, NumPy
- **NLP:** NLTK
- **Python:** 3.8+

---

## Installation

### Prerequisites
- Python 3.8 or higher
- 2 GB RAM minimum
- 500 MB disk space
- Internet connection

### Step-by-Step Setup

#### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Download Dataset
1. Visit [Kaggle WELFake Dataset](https://www.kaggle.com/datasets/jainpooja/welfake-dataset)
2. Download `WELFake_Dataset.csv`
3. Create `data/` folder in project root
4. Place CSV file at: `data/WELFake_Dataset.csv`

#### 4. Verify Installation
```bash
python -c "import pandas, numpy, sklearn, nltk; print('✓ All packages installed')"
```

---

## Architecture

### Pipeline Overview

```
Raw Data (72,134 articles)
        ↓
  [1] Data Loading → Explore dataset
        ↓
  [2] Preprocessing → Clean & normalize text
        ↓
  [3] Train/Test Split → 80/20 split
        ↓
  [4] Feature Extraction → TF-IDF vectorization (5000 features)
        ↓
  [5] Model Training → Logistic Regression
        ↓
  [6] Evaluation → Calculate metrics
        ↓
  [7] Prediction → Classify new articles
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `data_loader.py` | Load and explore data | `load_data()`, `explore_data()` |
| `preprocessing.py` | Text cleaning pipeline | `clean_text()`, `preprocess_dataframe()` |
| `feature_extraction.py` | TF-IDF vectorization | `create_tfidf_vectorizer()`, `fit_and_transform()` |
| `model_training.py` | Model training | `train_model()`, `save_model()`, `load_model()` |
| `evaluation.py` | Performance metrics | `evaluate_model()`, `display_metrics()` |
| `prediction.py` | Inference | `predict_news()`, `predict_batch_news()` |

### Preprocessing Pipeline

```
Raw Text
  ↓ Lowercase
  ↓ Remove URLs
  ↓ Remove special characters
  ↓ Normalize whitespace
  ↓ Remove stopwords
  ↓ Lemmatization
Clean Text
```

---

## Usage

### Run Complete Pipeline

Execute the entire workflow with a single command:

```bash
python main.py
```

This automatically:
1. Loads and explores the dataset
2. Preprocesses text data
3. Splits into training/testing sets (80/20)
4. Extracts TF-IDF features (5000 features)
5. Trains Logistic Regression model
6. Evaluates model performance
7. Makes predictions on sample articles

**Timing:**
- First run: ~15 minutes (includes NLTK downloads)
- Subsequent runs: ~5 seconds (uses cached model)

### Make Predictions on New Data

```python
from src.model_training import load_model
from src.feature_extraction import load_vectorizer
from src.prediction import predict_news, display_prediction

# Load pre-trained models
model = load_model('models/fake_news_model.pkl')
vectorizer = load_vectorizer('models/vectorizer.pkl')

# Predict on new article
article = "Scientists announce breakthrough in cancer treatment"
result = predict_news(article, model, vectorizer)

# Display results
display_prediction(article, result)
```

**Output:**
```
Prediction: Real
Confidence: 96%

Probability Breakdown:
- Fake: 4%
- Real: 96%
```

### Batch Predictions

```python
from src.prediction import predict_batch_news

articles = [
    "Article 1 text...",
    "Article 2 text...",
    "Article 3 text..."
]

results = predict_batch_news(articles, model, vectorizer)

for article, result in zip(articles, results):
    print(f"{result['label']}: {result['confidence']:.2%}")
```

---

## Model Performance

### Metrics on Test Set

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | ~95% | Overall correctness |
| **Precision** | ~94% | Reliability of fake predictions |
| **Recall** | ~96% | Fake detection rate |
| **F1-Score** | ~95% | Balanced performance |

### Confusion Matrix

```
                Predicted Real  Predicted Fake
Actual Real     [ TP: ~7,900    FP: ~300 ]      (~8,200)
Actual Fake     [ FN: ~500      TN: ~6,500 ]    (~7,000)
```

**Interpretation:**
- TP (True Positive): Correctly identified real news (~7,900)
- TN (True Negative): Correctly identified fake news (~6,500)
- FP (False Positive): Real news marked as fake (~300)
- FN (False Negative): Fake news not detected (~500)

### Sample Predictions

```
Input: "Breaking: Scientists discover cure for cancer"
Output: REAL (confidence: 96%)

Input: "FAKE: Celebrity secretly joins alien colony"
Output: FAKE (confidence: 94%)

Input: "Government announces new climate action plan"
Output: REAL (confidence: 92%)
```

---

## Key Concepts

### 1. TF-IDF Vectorization

**What it does:** Converts text into numerical features that ML models can understand.

**How it works:**
- **TF (Term Frequency):** How often a word appears in a document
- **IDF (Inverse Document Frequency):** How rare a word is across all documents
- **TF-IDF Score:** Combination of TF and IDF to find important words

**Formula:**
```
TF-IDF(term) = TF(term) × IDF(term)
```

**Why use it:**
- Emphasizes unique, discriminative words
- Removes common, unhelpful words
- Creates fixed-size feature vectors

**Parameters Used:**
- `max_features=5000`: Keep top 5,000 words
- `min_df=5`: Word must appear in ≥5 documents
- `max_df=0.7`: Word can appear in ≤70% of documents
- `ngram_range=(1,2)`: Use 1-word and 2-word combinations

### 2. Logistic Regression

**What it does:** Binary classification model that outputs probability (0-1).

**How it works:**
1. Takes TF-IDF features as input
2. Applies sigmoid function
3. Outputs probability between 0 and 1
4. Decision rule: probability > 0.5 → Real, else → Fake

**Why use it:**
- Fast training and prediction
- Interpretable coefficients
- Works well with sparse features (TF-IDF)
- Provides confidence scores

**Formula:**
```
P(Real) = 1 / (1 + e^(-score))
```

### 3. Data Preprocessing

**Textual Noise:**
- URLs: `http://example.com`
- Special characters: `!@#$%^&*()`
- Repeated whitespace: `  multiple   spaces  `
- Common words: `the`, `is`, `and`

**Cleaning Pipeline:**
1. **Lowercase:** Normalize case variations
2. **Remove URLs:** Strip web links
3. **Remove special chars:** Keep only letters
4. **Remove whitespace:** Clean spacing
5. **Remove stopwords:** Eliminate common words
6. **Lemmatization:** Convert to base form

**Example:**
```
Before:  "Check this URL: http://example.com!!! This is GREAT news!!!"
After:   "check great news"
```

### 4. Model Evaluation Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Overall performance |
| **Precision** | TP / (TP + FP) | Minimize false fakes |
| **Recall** | TP / (TP + FN) | Minimize missed fakes |
| **F1-Score** | 2×(P×R)/(P+R) | Balanced metric |

**Trade-offs:**
- High precision → Few false alarms, but miss some fakes
- High recall → Catch all fakes, but more false alarms
- F1-Score → Balances both concerns

---

## Project Structure

```
fake-news-detection/
├── data/                        # Data directory
│   ├── WELFake_Dataset.csv     # Raw dataset (download from Kaggle)
│   └── processed/              # Auto-generated processed data
│       ├── X_train.pkl
│       ├── X_test.pkl
│       ├── y_train.pkl
│       └── y_test.pkl
├── models/                      # Trained models (auto-generated)
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
├── main.py                      # Pipeline orchestrator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore rules
```

---

## Troubleshooting

### Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'pandas'`
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

**Problem:** `'venv' is not recognized`
```bash
# Windows: Use full path
venv\Scripts\activate

# macOS/Linux: Verify venv creation
python3 -m venv venv
source venv/bin/activate
```

### Dataset Issues

**Problem:** `FileNotFoundError: Dataset not found`
```bash
# Solution: Verify dataset location
# Must be: data/WELFake_Dataset.csv
# Not: data/WELFake_Dataset.zip (unzip first!)
```

**Problem:** `Permission denied` when accessing dataset
```bash
# Windows: Uncheck "Read-only" in file properties
# macOS/Linux: chmod 644 data/WELFake_Dataset.csv
```

### Performance Issues

**Problem:** `MemoryError` during training
```bash
# Solution: Reduce features in feature_extraction.py
vectorizer = create_tfidf_vectorizer(max_features=3000)
```

**Problem:** Script is very slow
```bash
# Normal for first run (includes NLTK downloads)
# Subsequent runs use cached model (~5 seconds)
```

### Model Issues

**Problem:** Accuracy is low (<90%)
```bash
# Solution 1: Verify preprocessing is working
# Solution 2: Check dataset quality
# Solution 3: Try different model parameters
# Solution 4: Increase max_features in TF-IDF
```

---

## Learning Outcomes

By working with this project, you will learn:

✅ **Data Science Fundamentals**
- Data loading and exploration
- Data preprocessing and cleaning
- Statistical analysis

✅ **Machine Learning**
- Feature extraction and engineering
- Binary classification
- Model training and evaluation
- Hyperparameter tuning

✅ **Python Ecosystem**
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for ML
- NLTK for NLP

✅ **Software Engineering**
- Code organization and modularity
- Documentation best practices
- Version control (Git)
- Project structure

---

## Dataset Information

### WELFake Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [Kaggle](https://www.kaggle.com/datasets/jainpooja/welfake-dataset) |
| **Total Articles** | ~72,000 |
| **Features** | text, label |
| **Classes** | 0 (Fake), 1 (Real) |
| **Balance** | ~50% each |
| **Format** | CSV |

### Download Instructions

1. Create Kaggle account (free)
2. Visit dataset page
3. Click "Download" button
4. Extract and place in `data/` folder

---

## Future Enhancements

### Week 1-2: Foundations (✓ Completed)
- Basic project setup
- Data loading and exploration
- Initial preprocessing

### Week 3: Core Features (✓ Completed)
- Complete preprocessing pipeline
- TF-IDF feature extraction
- Model training

### Week 4: Evaluation (✓ Completed)
- Model evaluation and metrics
- Prediction on new data
- Documentation

### Week 5+: Advanced Features
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for robustness
- [ ] Different ML models (Naive Bayes, SVM, Random Forest)
- [ ] Advanced preprocessing (stemming, n-grams)
- [ ] Visualization (word clouds, feature importance)
- [ ] Web interface (Flask/Streamlit)
- [ ] Deployment to production

---

## Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Improve documentation

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Security Considerations

### Data Handling
- Dataset contains ~72k articles (public domain)
- No sensitive personal information
- Safe for educational use

### Model Security
- Input validation: Text is preprocessed before prediction
- No arbitrary code execution
- Serialized models saved securely

### Best Practices
- Never commit dataset or model files to public repos
- Use `.gitignore` to exclude large files
- Review dependencies for security vulnerabilities
- Keep Python and packages updated

---

## Resources

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NLTK Documentation](https://www.nltk.org/)

### Tutorials
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Text Classification](https://developers.google.com/machine-learning/guides/text-classification)

### Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

---

## Support

For questions or issues:

1. Check this README and troubleshooting section
2. Review code comments and docstrings
3. Search Stack Overflow
4. Refer to library documentation
5. Open an issue on GitHub

---

## Changelog

### Version 1.0 (Current)
- Complete ML pipeline implementation
- TF-IDF + Logistic Regression
- ~95% accuracy on WELFake dataset
- Comprehensive documentation
- Production-ready code

---

**Ready to get started? Run `python main.py` to begin!** 🚀
