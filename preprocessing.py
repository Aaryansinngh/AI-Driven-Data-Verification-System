import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(
        df.select_dtypes(include=['int64', 'float64'])
    )

    return scaled_data
