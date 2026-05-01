import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Binarizer

def load_dataset(filename: str, drop_columns: list = None) -> pd.DataFrame:
    data = pd.read_csv(filename, delimiter=",")
    # Return the data without the `id` column (not useful)
    return data.drop(columns=drop_columns)

def feature_map_encoder(array: np.array, value_map: list) -> None:
    """
    Map the given array to the values in `value_map`
    This function overwrites the given `array`!
    """
    for i in range(len(array)):
        try:
            # Find the string in the map
            array[i] = value_map.index(array[i])
        except ValueError:
            raise Exception("Invalid value map")

def encode_feature(data: pd.DataFrame):
    """
    This function returns an array made of just numerical values,
    along with the map that's been used to achieve that.
    """
    # TODO: All of this might be already implemented in sklearn.preprocessing.OrdinalEncoder!
    # Take a look to the documentation of scikit-learn and see if it is simpler to handle.

    # Unique value map
    # The value map could also contain the occurencies of unique values (may be needed for some checks)
    # For each feature find all its possible values, if it is a categorical one, otherwise add an empty list to the map
    unique_values = [pd.unique(data.iloc[:, i]).tolist() if type(data.iloc[0,i]) == str else [] for i in range(len(data.columns))]

    # Convert into numpy array
    dataset = data.to_numpy()

    # Apply the map to the array
    for i in range(len(unique_values)):
        # Check if there's a mapping for this feature (empty list = numeric feature - not mapped)
        if len(unique_values[i]) > 0:
            feature_map_encoder(dataset.T[i], unique_values[i])
            dataset.T[i] = dataset.T[i].astype(int)

    # Return the encoded dataset (to float), along with the value map
    # (useful to map the features back to their original value)
    return dataset.astype(float), unique_values

def split_dataset(data: np.array):
    """
    Split the dataset into two arrays.
    The last column is the label set.
    Reshape the Y array into (n_samples, 1).
    """
    return data[:, :-1], data[:, -1].reshape(-1, 1).astype(int)

def filter_dataset(data: np.array) -> np.array:
    """
    Filter out records with missing values
    """
    return np.array([row for row in data if not np.isnan(row).any()])

def standardize(x: np.array) -> np.array:
    return StandardScaler().fit_transform(x)

def binarize(y: np.array) -> np.array:
    return Binarizer().fit_transform(y)