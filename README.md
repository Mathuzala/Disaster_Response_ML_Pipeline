# Disaster_Response_ML_Pipeline

## Purpose
The point of this projectis to analyze disaster data from the company Appen to build a model for an API that classifies disaster messages. There is a data set containing real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events to simulate a situation where someone can send the messages to an appropriate disaster relief agency. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

## ETL Pipeline (process_data.py)
Description:
The ETL (Extract, Transform, Load) process begins by reading in .csv files containing messages and categories of disaster response data. The two datasets are merged using a common 'id' and the categories are split into individual columns. After some cleaning (like converting category values into binary format and removing duplicates), the cleaned data is saved into a SQLite database.

## Machine Learning Pipeline (train_classifier.py)
Description:
This script reads in the data from the SQLite database and splits it into features (messages) and targets (categories). The text messages undergo a series of natural language processing steps (tokenization, vectorization, and TF-IDF transformation) to convert them into a format suitable for machine learning. A multi-output classifier using the RandomForest algorithm is trained on this processed data. Once trained the model's performance is evaluated on a test set and the model is saved to a .pkl file for later use in predicting message categories.

## Web App (run.py, master.html, go.html)
Description:
The Flask web application provides an interface where users can input a message and get classification results across different categories. The trained model is used to predict the categories of the message. The app also displays visualizations of the training data distribution.

### How to Launch the App:

    Ensure that all necessary Python libraries are installed.
    Navigate to the app's directory.
    Run the following command: python run.py
    Open a web browser and go to http://localhost:3001/ (or the port number provided in the terminal) to access the app.
