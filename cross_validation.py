"""K-Fold generator

Function created to divide a pandas DataFrame into multiple train - test
variations using the K-Fold method.

This function is dependant of the DataProcessing_CV file, used to
process the data.

This script requires that `pandas`, `numpy` and `sklearn` be installed
within the Python environment you are running this script in.

This file can be imported as a module and contains the following
function:

    * get_folds - returns an n-length array of train - test variations.
"""

from sklearn.model_selection import KFold
from DataProcessing_CV import DFrame, transformer
from config_file import k_folds
import pandas as pd


def get_folds(df, folds=k_folds):
    """Generates multiple train - test variations to use in cross-validation.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be divided into multiple train - test variations.
    folds : int, optional
        Used to define how many `folds` or train - test variations
        to generate. Predefined to 4.

    Returns
    -------
    df_folds
        a list of all train - test variations
    """

    splitter = KFold(n_splits=folds, shuffle=True, random_state=4294967295)

    # The needed df information is saved before turning them into numpy arrays.
    col_names = df.columns.tolist()
    dtypes_dict = {}
    for col in col_names:
        dtypes_dict[col] = df[col].dtypes

    # We divide and convert the feature columns and target column
    # into numpy arrays.
    df_features = df.iloc[:, :-1].values
    df_target = df[df.columns[-1]].values

    df_folds = []

    # For Loop to process and save each test - train variation
    for train_indices, test_indices in splitter.split(df_features):
        train_X, train_y = df_features[train_indices], df_target[train_indices]
        test_X, test_y = df_features[test_indices], df_target[test_indices]

        # We convert the numpy arrays back to pandas Dataframe.
        # Needed for DFrame processing.
        train_X = pd.DataFrame(train_X, columns=col_names[:-1])
        train_y = pd.DataFrame(train_y, columns=[col_names[-1]]) \
            .astype(dtypes_dict[col_names[-1]])[col_names[-1]]
        test_X = pd.DataFrame(test_X, columns=col_names[:-1])
        test_y = pd.DataFrame(test_y, columns=[col_names[-1]]) \
            .astype(dtypes_dict[col_names[-1]])[col_names[-1]]

        # Converting the columns back to its originals dtypes.
        for col in col_names[:-1]:
            train_X[col] = train_X[col].astype(dtypes_dict[col])
            test_X[col] = test_X[col].astype(dtypes_dict[col])

        # Processing and transforming the data.
        df_processor = DFrame(train_X, train_y, test_X, test_y)
        df_processor.process()
        train_X, train_y, test_X, test_y = df_processor.getAll()
        train_X, test_X = transformer(train_X, test_X, train_y)

        df_folds.append([train_X, train_y, test_X, test_y])

    return df_folds
