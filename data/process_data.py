import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This is the function that load the data from csv files
    
    Input
    messages_filepath - the path of the messages csv dataset
    categories_filepath - the path of the categories csv dataset
    
    Output
    Returns a dataframe with joined the 2 datasets
    """
    
    #Load messages
    messages = pd.read_csv(messages_filepath)
    #Load categories
    categories = pd.read_csv(categories_filepath)
    #Merge the 2 dataframes
    df = messages.merge(categories, left_on='id', right_on='id')

    return df


def clean_data(df):
    """
    This is the function that cleans the data of the joined df
    
    Input
    df - the df as we read it
    
    Output
    Returns a cleaned df
    """
    
    #Split categories into separate columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Rename the column names of categories
    row = categories.iloc[0]
    category_colnames = row.str.split(pat='-',expand=True)[0]
    categories.columns = category_colnames
    
    #Convert category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]
        categories[column] = categories[column].astype(int)
    categories.rename(columns={"1": "related"}, inplace = True)
    
    #Drop the original categories column from df
    df = df.drop(['categories'],axis=1)
    
    #Concatenate the original df with the new categories df
    df = df.merge(categories,left_index=True, right_index=True)
    
    #Remove duplicates
    df = df.drop_duplicates()
    
    #Remove the values equal to 2
    df = df[df['related']!=2]
    
    return df


def save_data(df, database_filename):
    """
    This is the function that loads the to a SQL lite DB and table
    
    Input
    df - the dataframe
    database_filename - the path and name of the DB
    
    Output
    No output
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    
    pass  


def main():
    """
    The main function that calls the rest ones
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
