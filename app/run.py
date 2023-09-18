import json
import plotly
import sqlite3
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

# load data
conn = sqlite3.connect('../DisasterResponse.db')
df = pd.read_sql("SELECT * FROM DisasterData", conn)
conn.close()

# load model
model = joblib.load("../classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()
    category_names = list(category_counts.index)
    
    # Create data for visualizing distribution of different categories
    category_distribution = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_distribution_names = list(category_distribution.index)

    # Data for 'Most Frequently Occurring Disaster Categories' pie chart visualization
    pie_category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False).head(10)
    pie_category_names = list(pie_category_counts.index)

    # create visuals
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
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                }
            }
        },
            {
            'data': [
                {
                    'type': 'pie',
                    'labels': pie_category_names,
                    'values': pie_category_counts
                }
            ],
            'layout': {
                'title': 'Top 10 Most Frequently Occurring Disaster Categories'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# data for 'Most Frequently Occurring Disaster Categories' pie chart visualization
category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False).head(10)
category_names = list(category_counts.index)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
