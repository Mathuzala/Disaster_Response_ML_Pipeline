
import sys
import sqlite3
import pandas as pd

messages_filepath = "./disaster_messages.csv"
categories_filepath = "./disaster_categories.csv'"

def load_data(messages_filepath, categories_filepath):
    """Load message and category data from csv files.
    
    Parameters:
    - messages_filepath: String, path to the messages csv file.
    - categories_filepath: String, path to the categories csv file.
    
    Returns:
    - DataFrame containing merged data from messages and categories.
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    return pd.merge(messages_df, categories_df, on='id')

def clean_data(df):
    """Process the combined message and category dataframe.
    
    - Splits categories into separate columns
    - Converts values to integers
    - Removes duplicates
    
    Parameters:
    - df: DataFrame, combined message and category data.
    
    Returns:
    - DataFrame, cleaned version of the input df.
    """
    # Spliting the 'categories' column from the merged_df into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Selecting the first row of the categories dataframe
    row = categories.iloc[0]

    # Using this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])
    
    # Renaming the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # Setting each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # Converting the column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Dropping the original categories column from `df`
    df = df.drop('categories', axis=1)

    # Concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Dropping duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """Save the processed data to a SQLite database.
    
    Parameters:
    - df: DataFrame, data to be saved.
    - database_filename: String, filename for the SQLite database.
    
    Returns:
    - None
    """
    conn = sqlite3.connect(database_filename)
    df.to_sql('DisasterData', conn, index=False, if_exists='replace')
    conn.close()

def main():
    """Main function to orchestrate data loading, processing, and saving.
    
    - Loads data from filepaths provided as command-line arguments
    - Processes the data
    - Saves the data to a SQLite database
    
    Returns:
    - None
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
        print('Please provide the filepaths of the messages and categories '              
              'datasets as the first and second argument respectively, as '              
              'well as the filepath of the database to save the cleaned data '              
              'to as the third argument. \n\nExample: python process_data.py '              
              'disaster_messages.csv disaster_categories.csv '              
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
