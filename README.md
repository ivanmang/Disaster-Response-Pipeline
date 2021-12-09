# Disaster-Response-Pipeline
## Installation
This repository was written in HTML and Python

All libraries are available in Anaconda distribution of Python

Python packages used:
`pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy`

## Project Overview
The goal of the project is to build a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages and integrated it into our web app.

## File Descriptions
1. `ETL Pipeline Preparation.ipynb`: The preparation of the development of ETL pipline 
2. `process_data.py` : An excutuble that read the csv files containing messages and categories, clean and merge the data and then stores it in a SQLite database
3. `ML Pipeline Preparation.ipynb` : The preparation of the development of a ML pipline 
4. `train_classifier.py` : An excutuble that read the data from SQL database, create and train an ML model and store it as a pickle file.
5. `data` : A folder contain categories dataset, messages dataset csv files, the database stored
6. `app` : A folder contain `run.py` that runs our webapp
7. `model`: A folder contain ML pipline and the ML model as pickle file
8. `templates`: A folder contain the html files for the webapp

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Display

<img src=https://user-images.githubusercontent.com/35868876/145328538-ec002132-2c5b-4d9d-a5a7-ccc89d677ac7.png width="800" >



<img src=https://user-images.githubusercontent.com/35868876/145328668-45829069-9df0-40be-8c3f-81cccd238c07.png width="800">
