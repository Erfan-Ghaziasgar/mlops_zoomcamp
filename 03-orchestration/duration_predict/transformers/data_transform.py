import pandas as pd

@transformer
def prepare_data(df, *args, **kwargs):
    """
    Prepare the data using the same logic as homework 1
    """
    # Calculate duration
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Filter duration between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical columns to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # Print the size after preparation (for Question 4)
    print(f"Size after data preparation: {len(df)}")
    
    return df