import pandas as pd


DATA_PATH = 'data/Music_Info.csv'

def clean_data(df):
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on 'spotify_id' column.
    2. Drops the 'spotify_id' and 'genre' column.
    3. Fill the missing values in the tags column with 'no_tags'.
    4. Converts the 'name', 'artist' and 'tags' column lowercase.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """

    return (
        df.drop_duplicates(subset='spotify_id')
        .drop(columns=['spotify_id', 'genre'])
        .fillna({'tags': 'no_tags'})
        .assign(
            name=lambda x: x['name'].str.lower(),
            artist=lambda x: x['artist'].str.lower(),
            tags=lambda x: x['tags'].str.lower()
        )
        .reset_index(drop=True)
    )


def data_for_content_filtering(data):
    """
    Cleans the input DataFrame by dropping specific columns.

    This function takes a DataFrame and removes the columns "track_id", "name",
    and "spotify_preview_url". It is intended to prepare the data for content based
    filtering by removing unnecessary features.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing songs information.

    Returns:
    pandas.DataFrame: A DataFrame with the specified columns removed.
    """
    return (
        data
        .drop(columns=["track_id","name","spotify_preview_url"])
    )
    

def main(data_path):
    """
    Main function to load, clean, and save data.
    Parameters:
    data_path (str): The file path to the raw data CSV file.
    Returns:
    None
    """
    # load the data
    data = pd.read_csv(data_path)
    
    # perform data cleaning
    cleaned_data = clean_data(data)
    
    # saved cleaned data
    cleaned_data.to_csv("data/cleaned_data.csv",index=False)
    

if __name__ == "__main__":
    main(DATA_PATH)