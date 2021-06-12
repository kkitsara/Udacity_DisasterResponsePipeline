In this folder you may find the below files.

README.md - The file that explains the repository purpose and structure

LICENSE - The license file

ETL Pipeline Preparation.ipynb - The jupyter notebook that includes the ETL of cleaning and preparing the data

process_data.py - The python code that does the same activities as the jupyter notebook for the ETL pipeline

ML Pipeline Preparation.ipynb - The jupyter notebook that includes the build of the ML pipeline and the storage of the model

train_classifier.py - The python code that does the same activities as the jupyter notebook for the ML pipeline

disaster_messages.7z - The dataset of the disaster messages zipped

disaster_categories.7z - The dataset that holds the disaster messages categories zipped

run.py - The python code that starts the site

The purpose of the project is to build an online classifier that accepts as input disaster messages and classifies them into different categories.
The project consists of the below parts:

ETL Pipeline Build
The ETL reads the 2 csv input files, joins them, cleans them accordingly and stores them to a SQL lite table in order to be used for the ML model training and testing
The jupyter notebook named ETL Pipeline Preparation.ipynb performs the steps in jupyter
The python code named process_data.py performs the steps in a python env

ML Pipeline Build
The ML reads the clean data from SQL lite, builds the ML pipeline, trains it, tests it and stores the results into a pickle file
The jupyter notebook named ML Pipeline Preparation.ipynb performs the steps in jupyter
The python code named train_classifier.py performs the steps in a python env

Web Site Start
The website starts by running in a python env the code run.py. In the site you can write as input a message and the site will revert the clasification result from 
the ML pipeline.

Below some info on how to run the python codes in a python ENV.


1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the folloDisasterResponsewing command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
