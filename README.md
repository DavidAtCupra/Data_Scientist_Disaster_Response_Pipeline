# 🌍 Disaster Response Pipeline Project

## 📘 Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation and Requirements](#installation-and-requirements)  
4. [Instructions](#instructions)  
   - [1️⃣ ETL Pipeline](#1️⃣-etl-pipeline)  
   - [2️⃣ Machine Learning Pipeline](#2️⃣-machine-learning-pipeline)  
   - [3️⃣ Flask Web App](#3️⃣-flask-web-app)  
5. [Data Visualizations](#data-visualizations)  
6. [Project Design and Components](#project-design-and-components)  
7. [Files and Code Description](#files-and-code-description)  
8. [Model Evaluation](#model-evaluation)  
9. [Improvements and Future Work](#improvements-and-future-work)  
10. [Author and Acknowledgements](#author-and-acknowledgements)

---

## 🧭 Project Overview

This project analyzes real messages sent during disaster events to build a **machine learning pipeline** that classifies messages into multiple categories (e.g., “water,” “shelter,” “medical help”).  

The goal is to help emergency services **route messages to the appropriate disaster relief agency** automatically.

The project includes:
- A full **ETL (Extract, Transform, Load)** data pipeline.
- A **Machine Learning model** trained for multi-output text classification.
- A **Flask web application** where users can input a new message and receive category predictions.
- Interactive **Plotly visualizations** describing the training dataset.

The dataset comes from [Appen](https://www.figure-eight.com) (formerly Figure 8).

---

## 🗂 Repository Structure

```
project_root/
│
├── app/
│   ├── run.py                     # Flask web app entry point
│   ├── static/
│   │   └── plotly-latest.min.js   # Local Plotly JS for offline mode
│   └── templates/
│       ├── master.html            # Main page with visualizations
│       └── go.html                # Classification result page
│
├── data/
│   ├── disaster_messages.csv      # Raw messages dataset
│   ├── disaster_categories.csv    # Raw categories dataset
│   ├── process_data.py            # ETL pipeline script
│   └── DisasterResponse.db        # Cleaned data (SQLite database)
│
├── models/
│   ├── train_classifier.py        # ML pipeline script
│   └── classifier.pkl             # Trained model (saved with joblib)
│
├── README.md                      # Documentation (this file)
└── requirements.txt               # Python dependencies
```

---

## ⚙️ Installation and Requirements

### Prerequisites
- plotly
- pandas
- nltk
- flask
- sqlalchemy
- scikit-learn

### Full requirements
- blinker==1.9.0
- click==8.3.0
- colorama==0.4.6
- Flask==3.1.2
- greenlet==3.2.4
- itsdangerous==2.2.0
- Jinja2==3.1.6
- joblib==1.5.2
- MarkupSafe==3.0.3
- narwhals==2.10.0
- nltk==3.9.2
- numpy==2.3.4
- packaging==25.0
- pandas==2.3.3
- plotly==6.3.1
- python-dateutil==2.9.0.post0
- pytz==2025.2
- regex==2025.10.23
- scikit-learn==1.7.2
- scipy==1.16.3
- six==1.17.0
- SQLAlchemy==2.0.44
- threadpoolctl==3.6.0
- tqdm==4.67.1
- typing_extensions==4.15.0
- tzdata==2025.2
- Werkzeug==3.1.3


### Install Required Packages
```bash
pip install -r requirements.txt
```

### NLTK Setup
You must download NLTK tokenizers and lemmatizers:
```python
import nltk
nltk.download(['punkt_tab', 'wordnet'])
```

---

## 🚀 Instructions

### 1️⃣ ETL Pipeline

**Goal:** Clean and prepare the dataset.

**Run:**
```bash
python .\data\process_data.py .\data\disaster_messages.csv .\data\disaster_categories.csv .\data\DisasterResponse.db
```

**What It Does:**
- Loads raw CSVs.
- Merges datasets on `id`.
- Splits the `categories` column into 36 binary category columns.
- Cleans missing values and duplicates.
- Saves clean data into a SQLite database (`DisasterResponse.db`).

---

### 2️⃣ Machine Learning Pipeline

**Goal:** Train and save the text classification model.

**Run:**
```bash
python .\models\train_classifier.py .\data\DisasterResponse.db classifier.pkl
```

**What It Does:**
- Loads clean data from SQLite.
- Splits into training and test sets.
- Builds a pipeline with:
  - `CountVectorizer` (custom NLTK tokenizer)
  - `TfidfTransformer`
  - `MultiOutputClassifier(RandomForestClassifier)`
- Tunes parameters using `GridSearchCV`.
- Evaluates precision, recall, and F1-score for each category.
- Saves the trained model as a pickle file (`classifier.pkl`).

---

### 3️⃣ Flask Web App

**Goal:** Visualize dataset and classify new messages.

**Run the web app:**
```bash
python app/run.py
```

Then open in your browser:
👉 [http://localhost:3001](http://localhost:3001)

**Features:**
- **Home page:** Interactive data visualizations via Plotly.  
- **Message input form:** Classifies custom disaster messages.  
- **Results page:** Displays predicted categories for the message.

---

## 📊 Data Visualizations

The web app includes three main visualizations (Plotly):

| # | Title | Description |
|---|--------|--------------|
| 1️⃣ | **Distribution of Message Genres** | Shows how many messages come from each communication channel (e.g., direct, news, social). |
| 2️⃣ | **Top 10 Most Frequent Disaster Categories** | Displays the most common disaster-related message types. |
| 3️⃣ | **Average Message Length by Genre** | Compares verbosity across genres (e.g., social vs. direct). |

---

## 🧩 Project Design and Components

### ETL Pipeline (`process_data.py`)
- Loads and merges raw CSV data.
- Cleans categories into binary columns.
- Removes duplicates and missing values.
- Saves cleaned DataFrame into SQLite database.

### ML Pipeline (`train_classifier.py`)
- Loads data from database.
- Tokenizes and lemmatizes text using **NLTK**.
- Transforms text using **CountVectorizer + TfidfTransformer**.
- Trains a **MultiOutput RandomForestClassifier**.
- Optimizes hyperparameters with Grid Search.
- Outputs evaluation metrics and saves final model as `classifier.pkl`.

### Flask Web App (`run.py`)
- Loads model and database.
- Renders interactive data visualizations (Plotly).
- Classifies user-provided messages.
- Displays prediction results clearly by category.

---

## 📈 Model Evaluation

- Evaluated using precision, recall, and F1-score for each category.
- Model performs best on frequent categories (e.g., *related*, *aid_related*).
- Some rare categories have lower recall due to data imbalance.
- Random Forest with TF-IDF achieved strong overall performance.

---

## 👩‍💻 Acknowledgements

**Project:** Disaster Response Pipeline  
**Course:** Udacity Data Scientist Nanodegree 

**Data source:**
- [Figure Eight](https://www.figure-eight.com) for providing the dataset.  

**Project source:**
- Udacity for project scaffolding and review criteria.

**Main libraries:**
- [Plotly](https://plotly.com/javascript/) and [Flask](https://flask.palletsprojects.com/) for visualization and deployment.  
