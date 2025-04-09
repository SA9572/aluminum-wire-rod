import pandas as pd

def load_data(file_path):
    """Load the CSV dataset into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Path to the dataset (relative to src/)
    data_path = "data/INT254_dataset_Final.csv"
    df = load_data(data_path)
    if df is not None:
        # Display basic info
        print("Dataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nDataset Shape:", df.shape)