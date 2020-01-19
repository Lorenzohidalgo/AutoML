"""Data processing functions

Set of functions used to process and transform the input DataSet.

This file can be imported as a module and contains the following
function:

    * transformer - Transforms the preprocessed data.
    * DFrame Class - Preprocesses the data.

"""

from systemtools.number import parseNumber
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
import category_encoders as ce
import collections
import numpy as np
import pandas as pd
import re


def transformer(train, test, train_y):
    """Scales and applies PCA to the input data.

    Parameters
    ----------
    train : DataFrame
        Features.
    test : DataFrame
        Features.
    test_y : numpy array
        Target.

    Returns
    -------
    train, test : numpy arrays
        Transformed Features.
    """
    train = train.dropna()
    test = test.dropna()
    scaler = RobustScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    clf = DecisionTreeClassifier()
    rfexv = RFECV(clf, cv=5)
    train = rfexv.fit_transform(train, train_y)
    test = rfexv.transform(test)
    return train, test


class DFrame:
    def __init__(self, trainF, trainT, testF, testT):
        """Init function.

        Parameters
        ----------
        trainF, testF : numpy arrays
            Features.
        trainT, TestT : numpy array
            Targets.
        """

        self.train = trainF
        self.train_target = trainT
        self.test = testF
        self.test_target = testT
        self.limit = ((len(trainF)+len(trainT))*0.01)
        self.idCols = []
        self.binaryDropped = []
        self.encDict = {}

    def __removeIdentifiersTrain(self):
        """Removes Identifiers.
            Drops all columns in the train Features that doesn't
            provide any usefull information. All changes are saved
            to be applied later to the test Features.
        """
        cols = self.train.select_dtypes(exclude=['float', 'float32',
                                                 'float64']).columns
        for col in cols:
            if len(self.train[col]) == len(self.train[col].unique()) \
                    or len(self.train[col].unique()) == 1:
                self.idCols.append(col)
                self.train = self.train.drop(columns=[col])
        cols = self.train.select_dtypes(exclude=['float', 'float32',
                                                 'float64']).columns
        for col in cols:
            if len(self.train[col].unique()) == 1:
                self.idCols.append(col)
                self.train = self.train.drop(columns=[col])
        self.objectCols = self.train.select_dtypes(exclude=['int', 'int32',
                                                            'int64', 'bool',
                                                            'float', 'float32',
                                                            'float64']) \
                                    .columns.tolist()

    def __removeIdentifiersTest(self):
        """Removes Identifiers.
            Drops all columns in the train Features that doesn't
            provide any usefull information with the same steps
            applied to the train set.
        """
        self.test = self.test.filter(items=np.setdiff1d(self.test.columns,
                                                        self.idCols).tolist())

    @staticmethod
    def __ordinalEncoding(col):
        """Applies Ordinal Encoding to a column

        Parameters
        ----------
        col : DataFrame
            Features.

        Returns
        -------
        encoder: ce.ordinalEncoder
            Fitted ordinal encoder.
        col: DataFrame
            Encoded Column
        """
        encoder = ce.OrdinalEncoder().fit(col)
        return encoder, encoder.transform(col)

    @staticmethod
    def __binaryEncodingTrain(col):
        """Applies binary Encoding to a column

        Parameters
        ----------
        col : DataFrame
            Features.

        Returns
        -------
        encoder: ce.binaryEncoder
            Fitted binary encoder.
        col: DataFrame
            Encoded Column
        """
        encoder = ce.BinaryEncoder().fit(col)
        dfbin = encoder.transform(col)
        header = []
        for col in dfbin.columns:
            if len(dfbin[col].unique()) == 1:
                header.append(col)
        return encoder, header, \
            dfbin.filter(items=np.setdiff1d(dfbin.columns, header).tolist())

    @staticmethod
    def __binaryEncodingTest(encoder, header, col):
        """Applies binary Encoding to a column with
        the train fitted encoder.

        Parameters
        ----------
        encoder: ce.binaryEncoder
            Fitted binary encoder.
        header: list
            List of column names to be dropped.
        col : DataFrame
            Features.

        Returns
        -------
        col: DataFrame
            Encoded Column
        """
        dfbin = encoder.transform(col)
        return dfbin.filter(items=np.setdiff1d(dfbin.columns, header).tolist())

    @staticmethod
    def __treatNumeric(col):
        """Formats the column numerically.

        Parameters
        ----------
        col : DataFrame
            Features.

        Returns
        -------
        col: DataFrame
            Encoded Column
        """
        df_temp = pd.to_numeric(col, errors='ignore', downcast='signed')
        if df_temp.isna().sum() > 0:
            df_temp = df_temp.fillna(df_temp.mean())
        return df_temp.apply(lambda x: parseNumber(x))

    def __treatTrain(self):
        """ Formats the Training set and saves the applied
        changes to execute them on the test set as well.
        """
        df1, df_temp = self.train[:], self.train[:]
        if self.train_target.dtype == 'O':
            self.encDict['targetCol'], self.train_target \
                = self.__ordinalEncoding(self.train_target)
        boolCols = df1.select_dtypes(include=['bool']).columns
        for col in boolCols:
            df1[col] = df1[col].astype(int)
        df_alpha = df1.select_dtypes(exclude=['int', 'int32', 'int64', 'bool',
                                              'float', 'float32', 'float64']) \
                      .applymap(lambda x: re.sub('[^A-Za-z]+', '', x), )
        for col in self.objectCols:
            alpha = df_alpha[col].str.isalpha()
            # Execution will prompt an error if comparison is made with is.
            if alpha.mask(alpha == False).isna().sum() <= self.limit:
                dropped = []
                self.encDict[col], dropped, dfbin \
                    = self.__binaryEncodingTrain(df1[col])
                df1 = df1.drop(columns=[col])
                df1 = pd.concat([dfbin, df1], axis=1)
                for drop in dropped:
                    self.binaryDropped.append(drop)
            else:
                df1[col] = self.__treatNumeric(df1[col])
        self.train = df1

    def __treatTest(self):
        """ Formats the test set following the same changes
        applied to the training set.
        """
        df1, df_temp = self.test[:], self.test[:]
        if self.test_target.dtype == 'O':
            self.test_target = self.encDict['targetCol'] \
                                   .transform(self.test_target)

        boolCols = df1.select_dtypes(include=['bool']).columns
        for col in boolCols:
            df1[col] = df1[col].astype(int)
        stringCols = np.setdiff1d(list(self.encDict.keys()),
                                  list(['targetCol'])).tolist()
        for col in stringCols:
            dfbin = self.__binaryEncodingTest(self.encDict[col],
                                              self.binaryDropped, df1[col])
            df1 = df1.drop(columns=[col])
            df1 = pd.concat([dfbin, df1], axis=1)

        cols = self.test.select_dtypes(exclude=['int', 'int32', 'int64',
                                                'bool', 'float', 'float32',
                                                'float64']).columns
        for col in np.setdiff1d(cols, stringCols).tolist():
            df1[col] = self.__treatNumeric(df1[col])
        self.test = df1

        # Correction of dtypes to match Train DataFrame
        for col in self.train.columns:
            self.test[col] = self.test[col].astype(self.train[col].dtypes)

        # Correction of column order to match Train DataFrame
        self.test = self.test[self.train.columns]

    @staticmethod
    def __isUnbalanced(array):
        """Checks if an array has balanced classes.

        Parameters
        ----------
        array : array
            Target array.

        Returns
        -------
        boolean: boolean
            True if unbalanced, else False.
        """
        n = len(array)
        classes = [(clas, float(count))
                   for clas, count in collections.Counter(array).items()]
        k = len(classes)
        H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])
        Hlog = H/np.log(k)
        if(Hlog < 0.995):
            return True
        else:
            return False

    @staticmethod
    def __turnBalanced(df):
        """Balances a unbalanced training set.

        Parameters
        ----------
        df : DataFrame
            Training set.

        Returns
        -------
        df_features: DataFrame
            Balanced features.
        df_target: DataFrame
            Balanced target.
        """
        dropCat = pd.DataFrame(df[df.columns[-1]].value_counts())
        if len(dropCat.index.tolist()) >= 10:
            limit = len(df)*0.05
        else:
            limit = 10
        dropCat = pd.DataFrame(df[df.columns[-1]].value_counts())
        dropCat = dropCat[dropCat[dropCat.columns[-1]] < limit].index.tolist()
        df = df[~df[df.columns[-1]].isin(dropCat)]
        df = df.dropna()
        df = df.reset_index()
        smt = SMOTETomek()
        X_smt, y_smt = smt.fit_sample(df.iloc[:, :-1], df[df.columns[-1]])
        collections.Counter(y_smt)
        df = pd.concat([pd.DataFrame(X_smt), pd.Series(y_smt)],
                       axis=1, sort=False)
        df_features = df.iloc[:, :-1]
        df_features = df_features.drop(columns=['index'])
        df_target = df[df.columns[-1]]
        return df_features, df_target

    def __treatUnbalanced(self):
        """ Checks and balances inbalanced trainin sets.
        """
        if self.__isUnbalanced(np.ravel(self.train_target)):
            train_df = pd.concat([pd.DataFrame(self.train),
                                  pd.Series(np.ravel(self.train_target))],
                                 axis=1, sort=False)
            self.train, self.train_target = self.__turnBalanced(train_df)

    def process(self):
        """ Call all the needed methods to preprocess the input dataset.
        """
        self.__removeIdentifiersTrain()
        self.__treatTrain()
        self.__removeIdentifiersTest()
        self.__treatTest()
        self.__treatUnbalanced()

    def trainFeatures(self):
        """ Returns train features.

        Returns
        -------
        self.train: DataFrame
            Preprocessed Training Features.
        """
        return self.train

    def trainTarget(self):
        """ Returns train targets.

        Returns
        -------
        self.train_target: array
            Preprocessed Training targets.
        """
        return np.ravel(self.train_target)

    def testFeatures(self):
        """ Returns test features.

        Returns
        -------
        self.test: DataFrame
            Preprocessed test Features.
        """
        return self.test

    def testTarget(self):
        """ Returns test targets.

        Returns
        -------
        self.test_target: array
            Preprocessed test targets.
        """
        return np.ravel(self.test_target)

    def getAll(self):
        """ Returns the divided and processed dataset.

        Returns
        -------
        self.train: DataFrame
            Preprocessed Training Features.
        self.train_target: array
            Preprocessed Training targets.
        self.test: DataFrame
            Preprocessed test Features.
        self.test_target: array
            Preprocessed test targets.
        """
        return self.train, np.ravel(self.train_target), \
            self.test, np.ravel(self.test_target)
