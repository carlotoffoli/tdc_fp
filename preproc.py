import numpy as np
import pandas as pd
from math import factorial, isnan
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
        try:
            # Find the string in the map
            array[i] = value_map.index(array[i])
        except ValueError:
            print(array[i], value_map)
            raise Exception("Invalid value map")

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
                if type(item) != str and isnan(item):
                    break
            if j < len(feature) - 1:
                # Found a nan, move it at the end
                feature[j] = feature[-1]
                feature[-1] = np.nan

        ordered_map.append(feature)

    return ordered_map

def get_value_map_sl(data: np.array):
    """
    Obtain a valid value map to use with OrdinalEncoder
    Ensure that all the np.nan values go to the end of each feature map
    """

    unordered_map = [pd.unique(col) for i, col in enumerate(data.T)]

    ordered_map = []

    for feature in unordered_map:
        for j, item in enumerate(feature):
            # Checking for nan in object arrays, this is the least problematic way
            if type(item) != str and isnan(item):
                break
        if j < feature.shape[0]-1:
            # Found a nan, move it at the end
            feature[j] = feature[-1]
            feature[-1] = np.nan

        ordered_map.append(feature)

    return ordered_map

def value_map_permutations_sl(value_map: list, encoded_features: list):
    """
    This function computes all the permutations of the given value map

    @param value_map: the value map provided by OrdinalEncoder
    @param encoded_features: list of encoded features indexes
    """

    # Devo costruire una value map che contiene una combinazione di permutazioni degli array delle features
    all_permutations = []
    for i, a in enumerate(value_map):
        if i in encoded_features:
            if a[-1] is np.nan:
                new = [list(i) + [a[-1],] for i in permutations(a[:-1])]
            else:
                new = permutations(a)
        else:
            new = [a,]
        all_permutations.append(new)

    return [list(combi) for combi in product(*all_permutations)]

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
            if a[-1] is np.nan:
                new = [list(perm) + [a[-1],] for perm in permutations(a[:-1])]
            else:
                new = permutations(a)
        else:
            new = [0,]
        all_permutations.append(new)

    return [list(combi) for combi in product(*all_permutations)]
        
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
            if category is np.nan: nan_categories.append(index)
            index += 1
    
    # could have used hstack
    encoded_data = np.append(np.delete(encoded_columns, nan_categories, axis=1), np.delete(data, categorical_features_indexes, axis=1), axis=1)

    encoded_feature_names = enc.get_feature_names_out([name for i, name in enumerate(feature_names) if i in categorical_features_indexes])
    encoded_feature_names = [name for i, name in enumerate(encoded_feature_names) if i not in nan_categories]
    encoded_feature_names += [name for i, name in enumerate(feature_names) if i in non_cat]

    return encoded_data.astype(float), encoded_feature_names

def encode_ordinal(data: np.array, value_map = []):
    """
    This function returns the ordinally encoded data 
    along with the value map used to translate it.
    """
    enc = OrdinalEncoder(categories=value_map if value_map else 'auto').fit(data)
    return enc.transform(data), enc.categories_

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
    from util import accuracy
    # from sklearn.metrics import accuracy_score as accuracy
    from clustering import kmeans

    encoded_features = [1,2,3,7,11] # TODO: Update when dropping features!
    csv_dataframe = load_dataset('tdc_fp/data/heart_disease_uci.csv', drop_columns=['id', 'ca', 'thal'])

    # dataset, feature_names = encode_onehot(csv_dataframe, encoded_features)
    # print(dataset.shape, feature_names)

    value_map = get_value_map(csv_dataframe.to_numpy(), encoded_features)

    # Random permutation code
    # value_map = [np.append(np.random.permutation(a[:-1]), a[-1]) if a[-1] is np.nan else np.random.permutation(a) if i in encoded_features else a for i, a in enumerate(enc.categories_)]
    
    # Check all the value map permutations
    best_score = 0
    for combination in value_map_permutations(value_map, encoded_features):
        dataset, _ = encode_dataset(csv_dataframe.to_numpy(), encoded_features, combination)
        feature_names = csv_dataframe.columns

        X, Y = split_dataset(filter_dataset(dataset, lambda x: not np.isnan(x[1]).any()))

        X = standardize(X)
        Y = binarize(Y).ravel()

        labels = kmeans(X)[0]
        accu = max(accuracy(labels, Y), 1-accuracy(labels, Y))
        if accu > best_score:
            best_score = accu
            best_map = combination
            print(best_score)
    print(best_score, best_map)