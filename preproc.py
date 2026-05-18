import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Binarizer, OneHotEncoder

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

def encode_dataset(dataframe: pd.DataFrame, categorical_features_indexes: list) -> np.array:
    """
    Dataset encoding through OneHotEncoder
    This function encodes the categorical features into multiple binary features and gets rid of nan data.
    Missing values are represented by all the binary features being set to 0.
    """
    feature_names = dataframe.columns
    data = dataframe.to_numpy()
    non_cat = np.delete(np.arange(data.shape[1]), categorical_features_indexes)
    enc = OneHotEncoder(sparse_output=False)
    encoded_columns = enc.fit_transform(np.delete(data, non_cat, axis=1))

    index = 0
    nan_categories = []
    for feature in enc.categories_:
        for category in feature:
            if category is np.nan: nan_categories.append(index)
            index += 1
    
    # could have used hstack
    encoded_data = np.append(np.delete(encoded_columns, nan_categories, axis=1), np.delete(data, categorical_features_indexes, axis=1), axis=1)

    encoded_feature_names = enc.get_feature_names_out([name for i, name in enumerate(feature_names) if i in categorical_features_indexes])
    encoded_feature_names = [name for i, name in enumerate(encoded_feature_names) if i not in nan_categories]
    encoded_feature_names += [name for i, name in enumerate(feature_names) if i in non_cat]

    return encoded_data, encoded_feature_names

def split_dataset(data: np.array):
    """
    Split the dataset into two arrays.
    The last column is the label set.
    Reshape the Y array into (n_samples, 1).
    """
    return data[:, :-1], data[:, -1].reshape(-1, 1).astype(int)

def filter_dataset(data: np.array, filter: callable) -> np.array:
    """
    This can be used to filter data.
    Data can be filtered either through columns or rows.
    That should be decided by transposing the array:
     - by rows: normal array
     - by cols: transposed array

    @param filter: must decide whether to add a row/col should be kept in data.
                   It must accept a tuple as input (row/col index, data array)
                   and return a bool.
    """
    return np.array([data[index] for index in range(data.shape[0]) if filter((index, data[index]))])

def standardize(x: np.array) -> np.array:
    return StandardScaler().fit_transform(x)

def binarize(y: np.array) -> np.array:
    return Binarizer().fit_transform(y)

if __name__ == '__main__':
    csv_dataframe = load_dataset('tdc_fp/data/heart_disease_uci.csv', drop_columns=['id'])

    dataset, feature_names = encode_dataset(csv_dataframe, [1,2,3,7,11,12,13])
    print(dataset.shape, feature_names)