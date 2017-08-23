
# coding: utf-8
# Author: Eliot Barril <eliot.barril@gmail.com>, Noémie Haouzi for indicator function
# License: BSD 3 clause


from indicator import indicator
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from collections import Counter

def get_mean(df, mode=False):
    res = []
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for var in list(df):
        if df[var].dtypes in numerics:
            res.append(df[var].mean())
        else:
            if mode==False:
                res.append(np.nan)
            else:
                res.append(Counter(df[var].dropna(axis=0)).most_common(1)[0][0])
    return res
    
def get_se(df):
    res = []
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for var in list(df):
        if df[var].dtypes in numerics:
            res.append(df[var].var())
        else:
            res.append(np.nan)
    return res

def get_min(df):
    res = []
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for var in list(df):
        if df[var].dtypes in numerics:
            res.append(df[var].min())
        else:
            res.append(np.nan)
    return res

def get_max(df):
    res = []
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for var in list(df):
        if df[var].dtypes in numerics:
            res.append(df[var].max())
        else:
            res.append(np.nan)
    return res

def indicator(df, mode=False):
    resume = pd.DataFrame(columns=['Variable', 'Type', 'Taux_NA',
                              'Nb_unique', 'Moyenne', 'Variance',
                              'Variance_normalisee', 'Min',
                              'Max'])
    resume['Variable'] = list(df)
    
    resume['Type'] = list(df.dtypes)
    resume['Taux_NA'] = list(df.isnull().sum()/df.shape[0])
    resume['Nb_unique'] = [df[var].nunique() for var in list(df)]
    resume['Moyenne'] = get_mean(df, mode=mode)
    resume['Variance'] = get_se(df)
    resume['Variance_normalisee'] = resume['Moyenne']/resume['Variance']
    resume['Min'] = get_min(df)
    resume['Max'] = get_max(df)

    return resume


class Features_keeper():
    """
    Selects useful features with less than threshold*100 percentage of NA
    Parameters
    ----------
    threshold : float between 0. and 1., defaut = 0.3
        The percentage of max TauxNA to consider
    """

    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.__fitOK = False
        self.__to_keep = []

    def fit(self, df_train, y_train=None):
        """
        Fits Features_keep.
        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with NA values
        y_train : pandas series of shape = (n_train, ) or None
            The target for regression task.
        Returns
        -------
        None
        """

        ### sanity checks
        if ((type(df_train)!=pd.SparseDataFrame)&(type(df_train)!=pd.DataFrame)):
            raise ValueError("df_train must be a DataFrame")

        if ((type(y_train) != pd.core.series.Series) &(type(y_train) != type(None))):
            raise ValueError("y_train must be a Series or None")

        
        indic = indicator(df_train)
        var_to_consider1 = list(indic[indic['Taux_NA']<self.threshold]['Variable'])
        var_to_consider2 = list(indic[indic['Nb_unique']!=1]['Variable'])
        var_to_consider = [val for val in var_to_consider1 if val in var_to_consider2]
        self.__to_keep = var_to_consider
        self.__fitOK = True

        return self

    def transform(self, df):
        """
        Transforms the dataset
        Drop the columns that have to many NAs
        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with NA
        Returns
        -------
        df : pandas dataframe of shape = (n_train, n_features2)
            The train dataset with relevant features
        """
        headers = list(df.columns.values)
        if(self.__fitOK):

            ### sanity checks
            if ((type(df)!=pd.SparseDataFrame)&(type(df)!=pd.DataFrame)):
                raise ValueError("df must be a DataFrame")
            for i in self.__to_keep:
            	if i not in headers:
            		raise ValueError(i + "must be in input's columns")

            return df[self.__to_keep]
        else:
            raise ValueError("call fit or fit_transform function before")

    def fit_transform(self, df_train, y_train=None):
        """
        Fits Features_keep and transforms the dataset
        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with NA
        y_train : pandas series of shape = (n_train, ) or None
            The target for regression task.
        Returns
        -------
        df_train : pandas dataframe
            Dataframe's shape = (n_train, n_features2)
            The train dataset with relevant features
        """

        self.fit(df_train, y_train)

        return self.transform(df_train)
