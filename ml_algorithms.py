"""Set of Machine Learning algorithms

Functions needed to execute diferent types of algorithms over
DataSets and to calculate the AUC result.ArithmeticError

This file can be imported as a module and contains the following
function:

    * run_RandomSearch - Returns best set of parameters from a
        given dictionary.
    * multiclass_RocAuc_Score - Calculates and returns the AUC.
    * getHighVariance - Returns the column indexes with a variance
        higher then the threshold.
    * deleteHighVariance - Deletes the columns.
    * automatedKNN - Executes k-nearest neighbors Algorithm.
    * automatedLogReg - Executes a Logistig regression.
    * automatedBerNB - Executes a Bernoulli Naive Bayes classifier.
    * automatedGaussNB - Executes a Gaussian naive Bayes classifier.
    * automatedPassiveAgr - Executes a Passive Aggressive Classifier.
    * automatedRidgeReg - Executes a Ridge Classifier.
    * automatedSGDReg - Executes a Stochastic Gradient Descent Classifier.
    * automatedSVM - Executes a Support-Vector Machine Classifier.
    * automatedDecisionTree - Executes a Decision Tree Classifier.
    * automatedRandomForest - Executes a Random Forest Classifier.
    * automatedBagging - Executes a Bagging Classifier.
    * automatedHistGB - Executes Histogram-based Gradient Boosting Classifier.
    * runMods - Executes all algorithms in a given list.
    * xVal_Means - Calculates the mean of the results for all K-Folds.
    * saveFinal - Saves all calculated results.

"""

from cross_validation import get_folds
import pandas as pd
from os import listdir, path
from numpy import logspace, unique
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, \
                                PassiveAggressiveClassifier, \
                                RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from func_timeout import func_timeout, FunctionTimedOut
from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, \
                                HistGradientBoostingClassifier
from numpy import ravel, geomspace, linspace, delete, var
from time import time
from collections import Counter
from commons import log_it, save_results
from timeout import timeout
from config_file import algorithm_timeout, search_cv, \
                        core_count, training_mdls


def run_RandomSearch(X, y, clf, param_grid, cv=search_cv):
    """Executes K-nearest neighbour on the given Data.

    Parameters
    ----------
    X : numpy arrays
        Features.
    y : numpy array
        Targets.
    clf : ****
        ******
    param_grid : *****
        Paramters to be evaluated.
    cv : ******
        Number of ****** predefined in the config_file.

    Returns
    -------
    random_search : ****
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: run_RandomSearch', 'Starting')
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_grid,
                                       scoring='roc_auc_ovr',
                                       n_jobs=core_count)
    random_search.fit(X, y)
    return random_search


def multiclass_RocAuc_Score(test_y, pred, average="macro"):
    """Calculates the AUC Score.

    Parameters
    ----------
    test_y : numpy arrays
        Test Target.
    pred : numpy array
        Predictions.
    average : string
        ******

    Returns
    -------
    roc_auc_score : float
        AUC score calculated by roc_auc_score.
    """

    log_it('Module: multiclass_RocAuc_Score', 'Starting')
    lb = LabelBinarizer()
    lb.fit(test_y)
    y_test = lb.transform(test_y)
    y_pred = lb.transform(pred)
    return roc_auc_score(y_test, y_pred, average=average,
                         multi_class="roc_auc_ovr")


def getHighVariance(array):
    """Returns columns with variance higher than the defined threshold.

    Parameters
    ----------
    array : numpy arrays
        Array to be evaluated.

    Returns
    -------
    delete_col : list
        List of column indexes to be deleted.
    """

    delete_col = []
    for col in range(0, array.shape[1]):
        if(var(array[:, col]) > 1000000):
            delete_col.append(col)
    delete_col.sort(reverse=True)
    return delete_col


def deleteHighVariance(array, selector):
    """Deletes columns from array defined by an index list..

    Parameters
    ----------
    array : numpy arrays
        Array to be modified.
    selector: list
        List of indexes to be deleted.

    Returns
    -------
    array : array
        Modified array.
    """
    for i in selector:
        array = delete(array, i, axis=1)
    return array


@timeout(algorithm_timeout)
def automatedKNN(train_X, train_y, test_X, test_y):
    """Executes K-nearest neighbour on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedKNN', 'Starting')
    k_max = round(train_y.shape[0] / len(Counter(train_y).keys())) * 0.8
    k_range = list(dict.fromkeys(geomspace(1, k_max, 50, dtype="int")))
    if(train_y.shape[0] > 200):
        if(train_X.shape[1] > 20):
            algorithm_sel = "ball_tree"
        else:
            algorithm_sel = "kd_tree"
    else:
        algorithm_sel = "auto"
    param_grid = {'n_neighbors': k_range, 'weights': ['uniform', 'distance']}
    model = KNeighborsClassifier(algorithm=algorithm_sel, n_jobs=core_count)
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


def automatedLogReg(train_X, train_y, test_X, test_y):
    """Executes Logistic Regression on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedLogReg', 'Starting')
    if(train_y.shape[0] > 200):
        if(train_X.shape[1] > 20):
            solver_sel = "sag"
        else:
            solver_sel = "saga"
    else:
        solver_sel = "liblinear"
    model = LogisticRegression(multi_class="ovr",
                               solver=solver_sel,
                               n_jobs=core_count)
    param_grid = {'C': logspace(-3, 3, 7), 'penalty': ["l1", "l2"]}
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


def automatedBerNB(train_X, train_y, test_X, test_y):
    """Executes Bernoulli Naive Bayes classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedBerNB', 'Starting')
    model = BernoulliNB()
    param_grid = {'alpha': linspace(0.1, 1, 10)}
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


def automatedGaussNB(train_X, train_y, test_X, test_y):
    """Executes Gaussian naive Bayes classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedGaussNB', 'Starting')
    model = GaussianNB()
    model.fit(train_X, ravel(train_y))
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


def automatedPassiveAgr(train_X, train_y, test_X, test_y):
    """Executes Passive Aggressive Classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedPassiveAgr', 'Starting')
    model = PassiveAggressiveClassifier(fit_intercept=True, n_jobs=core_count)
    model = model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


def automatedRidgeReg(train_X, train_y, test_X, test_y):
    """Executes Ridge Classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedRidgeReg', 'Starting')
    param_grid = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0]}
    model = RidgeClassifier(fit_intercept=False)
    model = GridSearchCV(estimator=model, param_grid=param_grid)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


def automatedSGDReg(train_X, train_y, test_X, test_y):
    """Executes Stochastic Gradient Descent Classifier.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedSGDReg', 'Starting')
    param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                  'max_iter': [1000],
                  'loss': ['log', 'modified_huber'],
                  'penalty': ['l1', 'l2'],
                  'n_jobs': [-1]}
    model = SGDClassifier(n_jobs=core_count)
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


@timeout(algorithm_timeout)
def automatedSVM(train_X, train_y, test_X, test_y):
    """Executes Support-Vector Machine Classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    selector = getHighVariance(train_X)
    train_X = deleteHighVariance(train_X, selector)
    if(train_X.shape[1] <= 2):
        log_it('Module: Bagging automatedSVM',
               'Skipped, Shape:'+str(train_X.shape))
        return 0
    elif(train_y.shape[0] > 200 & (train_X.shape[1] > 20)):
        log_it('Module: Bagging automatedSVM', 'Starting')
        param_grid = [{'base_estimator__kernel': ['rbf'],
                       'base_estimator__gamma': [1e-2, 1e-3, 1e-5],
                       'base_estimator__C': [0.1, 1, 10, 50],
                       'n_estimators': [40, 100],
                       'max_samples': [0.1, 0.2, 0.5]},
                      {'base_estimator__kernel': ['sigmoid'],
                       'base_estimator__gamma': [1e-2, 1e-3,  1e-5],
                       'base_estimator__C': [0.1, 1, 10, 50],
                       'n_estimators': [40, 100],
                       'max_samples': [0.1, 0.2, 0.5]},
                      {'base_estimator__kernel': ['linear'],
                       'base_estimator__C': [0.1, 1, 10, 50],
                       'n_estimators': [40, 100],
                       'max_samples': [0.1, 0.2, 0.5]},
                      {'base_estimator__kernel': ['poly'],
                       'base_estimator__degree': [2, 3, 4],
                       'base_estimator__C': [0.1, 1, 10, 50],
                       'n_estimators': [40, 100],
                       'max_samples': [0.1, 0.2, 0.5]}]
        model = BaggingClassifier(base_estimator=SVC(), n_jobs=core_count)
        model = run_RandomSearch(train_X, train_y, model, param_grid)
    else:
        log_it('Module: automatedSVM', 'Starting')
        param_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3,  1e-5],
                       'C': [0.1, 1, 10, 50]},
                      {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3,  1e-5],
                       'C': [0.001, 0.10,  10, 50]},
                      {'kernel': ['linear'], 'C': [0.1, 1, 10, 50]},
                      {'kernel': ['poly'], 'degree': [2, 3, 4],
                       'C': [0.1, 1, 10, 50]}]
        model = SVC(decision_function_shape='ovr', probability=True)
        model = run_RandomSearch(train_X, train_y, model, param_grid)
    test_X = deleteHighVariance(test_X, selector)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


@timeout(algorithm_timeout)
def automatedDecisionTree(train_X, train_y, test_X, test_y):
    """Executes Decision Tree Classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedDecisionTree', 'Starting')
    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_split": geomspace(2, 50, 10, dtype=int),
                  "max_depth": geomspace(2, 50, 8, dtype=int),
                  "min_samples_leaf": geomspace(2, 50, 8, dtype=int),
                  "max_leaf_nodes": geomspace(2, 200, 10, dtype=int)}
    model = DecisionTreeClassifier()
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


@timeout(algorithm_timeout)
def automatedRandomForest(train_X, train_y, test_X, test_y):
    """Executes Random Forest Classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedRandomForest', 'Starting')
    param_grid = {'bootstrap': [True],
                  'max_depth': [50, 100, 200],
                  'min_samples_leaf': [2, 4, 8],
                  'min_samples_split': [2, 4, 8, 12],
                  'n_estimators': [100, 500, 1000]}
    model = RandomForestClassifier()
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


@timeout(algorithm_timeout)
def automatedBagging(train_X, train_y, test_X, test_y):
    """Executes Bagging Classifier on the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedBagging', 'Starting')
    param_grid = {'n_estimators': [40, 100],
                  'base_estimator__max_depth': [4, 5, 6],
                  'base_estimator__max_leaf_nodes': [10, 25],
                  'max_samples': [0.05, 0.1, 0.2, 0.5]}
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                              n_jobs=core_count)
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


@timeout(algorithm_timeout)
def automatedHistGB(train_X, train_y, test_X, test_y):
    """Executes Histogram-based Gradient Boosting Classifier.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.

    Returns
    -------
    multiclass_RocAuc_Score: float
        AUC score calculated by multiclass_RocAuc_Score.
    """

    log_it('Module: automatedHistGB', 'Starting')
    param_grid = {'max_iter': [1000, 1200, 1500],
                  'learning_rate': [0.1],
                  'max_depth': [25, 50, 75]}
    model = HistGradientBoostingClassifier()
    model = run_RandomSearch(train_X, train_y, model, param_grid)
    pred = model.predict(test_X)
    return multiclass_RocAuc_Score(test_y, pred)


def runMods(train_X, train_y, test_X, test_y,
            list_of_mdls=training_mdls):
    """Executes all algorithms in the list for the given Data.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.
    list_of_mdls: list, optional
        List with the function names to be executed.
        Default set to all algorithms.

    Returns
    -------
    Results: list
        List with the results of each runned function.
    """

    log_it('Module: runAllMod', 'Starting')
    Results = pd.DataFrame(columns=['AUC', 'time'])
    for mdl in list_of_mdls:
        start = time()
        row = pd.Series([eval(mdl + '(train_X, train_y, test_X, test_y)'),
                         time()-start], index=['AUC', 'time'], name=mdl)
        Results = Results.append(row)
    return Results


def xVal_Means(df1, df2, df3, df4):
    """Executes Logistic Regression on the given Data.

    Parameters
    ----------
    df_* : DataFrame
        DataFrame with the results of each K-Fold.

    Returns
    -------
    result: DataFrame
        DataFrame with the calculated mean of the results.
    """

    dfs = [df1, df2, df3, df4]
    return pd.concat([each.stack() for each in dfs], axis=1) \
             .apply(lambda x: x.mean(), axis=1).unstack()


def saveFinal(train_X, test_X, train_y, df_res, filename):
    """Saves the calculated results by calling the save_results function.

    Parameters
    ----------
    train_X, test_X : numpy arrays
        Train and test Features.
    train_y, test_y : numpy array
        Train and test Targets.
    df_res: DataFrame
        DataFrame containing the calculated results.
    filename: string
        Name of the analyzed DataSet.
    """

    log_it('DataSet: ' + filename, 'Saving results')
    df_res['algorithm'] = df_res.index
    df_res['dataset'] = filename
    df_res['num_rows'] = train_X.shape[0] + test_X.shape[0]
    df_res['num_vars'] = train_X.shape[1]
    df_res['num_clases'] = len(unique(train_y))
    df_res = df_res.sort_values(['AUC', 'time'], ascending=[0, 1])
    df_res.apply(lambda row: save_results(row['dataset'], row['num_rows'],
                                          row['num_vars'], row['num_clases'],
                                          row['algorithm'], row['AUC'],
                                          row['time']), axis=1)
