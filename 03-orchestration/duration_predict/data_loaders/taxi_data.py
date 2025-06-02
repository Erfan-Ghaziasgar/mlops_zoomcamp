import pandas as pd

@data_loader
def load_data(*args, **kwargs):
    """
    Load NYC Yellow taxi data for March 2023
    """
    # URL for March 2023 Yellow taxi data
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    
    df = pd.read_parquet(url)
    
    # Print the number of records (for Question 3)
    print(f"Number of records loaded: {len(df)}")
    
    return df