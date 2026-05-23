import numpy as np
import pandas as pd
from math import isnan
from itertools import permutations, product
from sklearn.preprocessing import StandardScaler, Binarizer, OneHotEncoder, OrdinalEncoder

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
        if type(array[i]) == float and np.isnan(array[i]) and np.isnan(value_map[-1]): 
            # The value map accounts for nans and is ordered correctly
            array[i] = len(value_map)
        else:
            try:
                # Find the string in the map
                array[i] = value_map.index(array[i])
            except ValueError:
                print(type(array[i]), np.isnan(array[i]))
                raise Exception(f"Invalid value map: {array[i]} not in map {value_map}")

def encode_dataset(data: np.array, encoded_features: list, value_map: list = None):
    """
    This function returns an array made of just numerical values,
    along with the map that's been used to achieve that.
    """
    # Unique value map
    # The value map could also contain the occurencies of unique values (may be needed for some checks)
    # For each feature find all its possible values, if it is a categorical one, otherwise add an empty list to the map
    unique_values = value_map if value_map else get_value_map(data, encoded_features)

    # Apply the map to the array
    for i, feature in enumerate(unique_values):
        # Check if there's a mapping for this feature (zero = non-categorical feature)
        if feature != 0:
            feature_map_encoder(data.T[i], feature)
            data.T[i] = data.T[i].astype(int)

    # Return the encoded dataset (to float), along with the value map
    # (useful to map the features back to their original value)
    return data.astype(float), unique_values

def get_value_map(data: np.array, encoded_features: list):
    """
    This function calculates the value map to use with encode_dataset
    """

    unordered_map = [pd.unique(col).tolist() if i in encoded_features else 0 for i, col in enumerate(data.T)]

    ordered_map = []

    for feature in unordered_map:
        if feature != 0: # feature in encoded_features
            # Checking for nan in object arrays, this is the least problematic way
            for j, item in enumerate(feature):
                if type(item) == float and np.isnan(item):
                    break
            if j < len(feature) - 1:
                # Found a nan, move it at the end
                feature[j] = feature[-1]
                feature[-1] = np.nan

        ordered_map.append(feature)

    return ordered_map

def value_map_permutations(value_map: list, encoded_features: list):
    """
    This function computes all the permutations of the given value map

    @param value_map: the value map provided by get_value_map
    @param encoded_features: list of encoded features indexes
    """

    # Devo costruire una value map che contiene una combinazione di permutazioni degli array delle features
    all_permutations = []
    for i, a in enumerate(value_map):
        if i in encoded_features:
            if type(a[-1]) == float and np.isnan(a[-1]):
                new = [list(perm) + [a[-1],] for perm in permutations(a[:-1])]
            else:
                new = permutations(a)
        else:
            new = [0,]
        all_permutations.append(new)

    return product(*all_permutations)
        
def encode_onehot(dataframe: pd.DataFrame, categorical_features_indexes: list) -> np.array:
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
            # This time nans were not a problem as they're all type float
            if category is np.nan: nan_categories.append(index)
            index += 1
    
    # could have used hstack
    encoded_data = np.append(np.delete(encoded_columns, nan_categories, axis=1), np.delete(data, categorical_features_indexes, axis=1), axis=1)

    encoded_feature_names = enc.get_feature_names_out([name for i, name in enumerate(feature_names) if i in categorical_features_indexes])
    encoded_feature_names = [name for i, name in enumerate(encoded_feature_names) if i not in nan_categories]
    encoded_feature_names += [name for i, name in enumerate(feature_names) if i in non_cat]

    return encoded_data.astype(float), encoded_feature_names

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
    return np.array([array for index, array in enumerate(data) if filter((index, array))])

def binarize(y: np.array) -> np.array:
    return Binarizer().fit_transform(y)

class Preprocess:
    def __init__(self, dropped = [], encoded_features = [1,2,3,7,11,12,13], ordinal = True, permute_map = False, multiclass = False, permute_data = False, random_state = None):
        """
        Preprocessing Class
        
        If you drop any cols with dropped remember to shift the encoded_features array!
        """

        self._csv_dataframe = load_dataset('data/heart_disease_uci.csv', drop_columns=['id',] + dropped)

        self._encoded_features = encoded_features

        if random_state is not None:
            self.seed = random_state
            np.random.seed(self.seed)

        # Encode categorical features as indexed in encoded_features
        if ordinal:
            self.value_map = get_value_map(self._csv_dataframe.to_numpy(), self._encoded_features)
            if permute_map:
                new_map = []
                for i, a in enumerate(self.value_map):
                    if i in self._encoded_features:
                        if type(a[-1]) == float and np.isnan(a[-1]):
                            # Also this time, nans gave us headaches!
                            n = np.random.permutation(a[:-1]).tolist() + [np.nan,]
                        else: 
                            n = np.random.permutation(a).tolist()
                    else: 
                        n = 0
                    new_map.append(n)
                self.value_map = new_map
            self._dataset, _ = encode_dataset(self._csv_dataframe.to_numpy(), self._encoded_features, self.value_map)
            self.feature_names = self._csv_dataframe.columns
        else:
            self._dataset, self.feature_names = encode_onehot(self._csv_dataframe, self._encoded_features)

        self.X, self.Y = split_dataset(filter_dataset(self._dataset, lambda x: not np.isnan(x[1]).any()))

        self._standardize()
        if not multiclass:
            self.Y = binarize(self.Y)

        if permute_data:
            indexes = np.random.permutation(self.X.shape[0])
            self.X = self.X[indexes]
            self.Y = self.Y[indexes]

        print("Dataset is", self.X.dtype, self.X.shape)

        # Flatten Y
        self.Y = self.Y.ravel()

    def _standardize(self):
        self._scaler = StandardScaler().fit(self.X)
        self.X = self._scaler.transform(self.X)

    def rescale(self):
        """
        Revert the standardization over X for representation
        """
        return self._scaler.inverse_transform(self.X)