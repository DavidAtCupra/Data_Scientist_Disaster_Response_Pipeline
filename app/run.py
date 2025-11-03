import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine
import joblib

# pip install flask 
# pip install plotly 

app = Flask(__name__)


def tokenize(text):
    """Tokenize and lemmatize input text."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


# Load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
model = joblib.load("./models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    """Main page: displays visualizations of the dataset."""

    # --- Visualization 1: Message genre distribution ---
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # --- Visualization 2: Top 10 most frequent categories ---
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    top_categories = category_counts.head(10)

    # --- Visualization 3: Proportion of related vs. not related ---
    related_counts = df['related'].value_counts()
    related_labels = ['Not Related', 'Related']

    # Create Plotly graphs
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=top_categories.index,
                    y=top_categories.values
                )
            ],
            'layout': {
                'title': 'Top 10 Most Frequent Disaster Categories',
                'yaxis': {'title': "Message Count"},
                'xaxis': {'title': "Category", 'tickangle': -45}
            }
        },
        {
            'data': [
                Pie(
                    labels=related_labels,
                    values=related_counts
                )
            ],
            'layout': {
                'title': 'Proportion of Related vs. Not Related Messages'
            }
        }
    ]

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """Handles user query and displays model prediction results."""
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html template with results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Run the Flask web server."""
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
