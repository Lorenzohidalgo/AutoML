"""Auto Machine Learning

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

from ml_algorithms import runMods, xVal_Means
from commons import log_it
from timeout import timeout
from cross_validation import get_folds
from config_file import automl_timeout, automl_explicit, \
                        default_algorithm, model_path, trns_primitives
from DataProcessing_CV import transformer
import numpy as np
import pandas as pd
import featuretools as ft
import pickle

# Global result dictionary
resultsDict = {}
decission_tree = pickle.load(open(model_path, 'rb'))


def save_result(algorithm_name, AUC):
    """Saves the results into the dictionary.
    Only the highest AUC for each algorithm is kept.

    Parameters
    ----------
    algorithm_name : string
        Name of the used algorithm.
    AUC : float
        Calculated AUC for the given algorithm.
    """
    global resultsDict
    if algorithm_name in resultsDict.keys():
        if resultsDict[algorithm_name] < AUC:
            resultsDict[algorithm_name] = AUC
    else:
        resultsDict[algorithm_name] = AUC


def average(num_list):
    """Returns the average value of a numeric list.

    Parameters
    ----------
    num_list : list
        Numeric list.

    Returns
    -------
    average : float
        Average value of the numeric list.
    """
    return sum(num_list)/len(num_list)


def get_foldFeatures(folds):
    """Extracts and returns the specific DataSet Features.

    Parameters
    ----------
    folds : array
        Array containing the diferent DataSet folds.

    Returns
    -------
    array : array
        Array containing the extracted DataSet features.
    """
    rows = []
    classes = []
    ints = []
    floats = []
    cols = []
    for fold_number in range(len(folds)):
        X, Y = folds[fold_number][:2]
        classes.append(len(np.unique(Y)))
        df = pd.DataFrame(data=X)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='signed')
        rows.append(len(Y))
        ints.append(len(df.select_dtypes(exclude=['float', 'float32',
                                                  'float64']).columns))
        floats.append(len(df.columns) - ints[-1])
        cols.append(ints[-1] + floats[-1])
    return np.asarray([average(rows), average(classes), average(ints),
                       average(floats), average(cols)])


def generateFeatures(train_X, test_X, counter):
    """Generates new features for a given fold

    Parameters
    ----------
    train_X : array
        Train Features.
    test_X : array
        Test Features.
    counter : int
        Number to specify the type of feature generation to be used.

    Returns
    -------
    average : float
        Average value of the numeric list.
    """
    es1 = ft.EntitySet()
    es2 = ft.EntitySet()
    df1 = pd.DataFrame(train_X, columns=[str(i).zfill(2)
                                         for i in range(0, train_X.shape[1])])
    df2 = pd.DataFrame(test_X, columns=[str(i).zfill(2)
                                        for i in range(0, test_X.shape[1])])
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    es1.entity_from_dataframe(entity_id='data', dataframe=df1, index='index')
    es2.entity_from_dataframe(entity_id='data', dataframe=df2, index='index')
    feature_matrix1, feature_defs1 = ft.dfs(entityset=es1,
                                            target_entity='data',
                                            trans_primitives=trns_primitives[counter])
    feature_matrix2, feature_defs2 = ft.dfs(entityset=es2,
                                            target_entity='data',
                                            trans_primitives=trns_primitives[counter])
    return feature_matrix1, feature_matrix2


def format_df(df):
    """Formats a Data Frame to avoid possible execution errors.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    df : DataFrame
        Output DataFrame.
    """
    df = df.reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast='signed')
        df[col][df[col] == np.inf] = np.nan
        df[col].fillna(df[col].mean(), inplace=True)
    return df


def add_features(folds, counter):
    """Controls the addition of new features to every fold.

    Parameters
    ----------
    folds : array
        Array containing the diferent DataSet folds.
    counter : int
        Number to specify the type of feature generation to be used.

    Returns
    -------
    returning_folds : array
        Array containing the diferent DataSet folds.
    """
    log_it('Module: add_features', 'Adding Features.')
    resulting_folds = []
    for train_X, train_y, test_X, test_y in folds:
        try:
            new_train_X, new_test_X = generateFeatures(train_X,
                                                       test_X, counter)
            new_train_X = format_df(new_train_X)
            new_test_X = format_df(new_test_X)
            new_train_X, new_test_X = transformer(new_train_X,
                                                  new_test_X, train_y)
            resulting_folds.append([new_train_X, train_y, new_test_X, test_y])
        except:
            log_it('Module: add_features', 'Generation skipped.')
            resulting_folds.append([train_X, train_y, test_X, test_y])
    log_it('Module: add_features', 'Returning new folds.')
    return resulting_folds


@timeout(automl_timeout)
def evaluate(df):
    """Function to execute models and generating new features
    until AUC == 1 or the maximum time has elapsed.

    Parameters
    ----------
    df: DataFrame
        DataFrame to be evaluated.
    """
    log_it('Module: evaluate', 'Starting.')
    folds = get_folds(df)
    predicted_algorithm = default_algorithm
    iter_control = -1

    while True:
        count = 1
        for train_X, train_y, test_X, test_y in folds:
            globals()['results%s' % count] = runMods(train_X,
                                                     train_y,
                                                     test_X,
                                                     test_y,
                                                     predicted_algorithm)
            count += 1
        df_res = xVal_Means(results1, results2, results3, results4)
        df_res['algorithm'] = df_res.index
        df_res.apply(lambda row: save_result(row['algorithm'],
                                             row['AUC']), axis=1)
        log_it('Module: evaluate', 'Iteration: ' + str(iter_control) +
               ' Result: ' + str(df_res.iloc[0]['algorithm']) + ' : ' +
               str(df_res.iloc[0]['AUC']))
        if max(list(resultsDict.values())) == 1.0 or iter_control > 6:
            log_it('Module: evaluate', 'Exits, Iteration: ' +
                   str(iter_control) + ' AUC: ' +
                   str(max(list(resultsDict.values()))))
            break
        else:
            if iter_control >= 0:
                folds = add_features(folds, iter_control)
            folds_features = get_foldFeatures(folds).reshape(1, -1)
            if folds_features[0][-1] == 1.0:
                log_it('Module: evaluate', 'Exits, only one column left.')
                break
            predicted_algorithm = decission_tree.predict(folds_features) \
                .tolist()
            iter_control += 1
            log_it('Module: evaluate', 'Predicted: ' +
                   str(predicted_algorithm[0]) +
                   ' For iteration: ' + str(iter_control))


def automl(df, explicit=automl_explicit):
    """Main Function.

    Parameters
    ----------
    df: DataFrame
        DataFrame to be evaluated.
    explicit: boolean, optional
        If True it will also return the algorithm name.

    Returns
    -------
    AUC: float
        Best AUC archived.
    algorithm_name: string, optional
        Name of the best performing algorithm.
    """
    global resultsDict
    log_it('Module: automl', 'Starting.')
    evaluate(df)
    if explicit:
        return max(resultsDict, key=resultsDict.get), \
            resultsDict[max(resultsDict, key=resultsDict.get)]
    else:
        return max(list(resultsDict.values()))
