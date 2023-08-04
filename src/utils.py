from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import logging
import sys
import os
import yaml
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def load_config(config_name):
    logger.info('Loading Config File')
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

class ScalerDf(BaseEstimator, TransformerMixin):
    """A custom transformer that applies scaling to a pandas DataFrame.
    
    Parameters
    ----------
    method : str
        The scaling method to be applied. Supported values are 'minmax', 'standard', and 'none'.
    
    Attributes
    ----------
    method : str
        The scaling method to be applied.
        
    Methods
    -------
    transform(X, y=None)
        Apply the specified scaling method to the input DataFrame.
    fit(X, y=None)
        Fit the scaler to the input DataFrame.
    """
    def __init__(self, method):
        self.method = method

    def transform(self,X,y=None):
        if self.method == 'minmax':
            scaler = MinMaxScaler()
        elif self.method == 'standard':
            scaler = StandardScaler()
        elif self.method == 'none': # Agregar condición para no hacer nada
            return X
        scaler.fit(X)
        X = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index = X.index
            )
        return X

    def fit(self, X, y=None):
        return self
    
class Kmeans_(BaseEstimator, TransformerMixin):
    """A custom transformer that applies K-means clustering to a pandas DataFrame.
    
    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters to form. Default is 2.
    n_init : int, optional
        Number of times the K-means algorithm will be run with different centroid seeds. Default is 10.
    
    Attributes
    ----------
    n_clusters : int
        The number of clusters to form.
    n_init : int
        Number of times the K-means algorithm will be run with different centroid seeds.
        
    Methods
    -------
    transform(X, y=None)
        Apply the fitted K-means model to the input DataFrame and add a 'label' column containing the predicted cluster labels.
    fit(X, y=None)
        Fit the K-means model to the input DataFrame.
    """
    def __init__(self, n_clusters = 2, n_init = 10):
        self.n_clusters = n_clusters
        self.n_init = n_init

    def transform(self, X, y=None):
        labels = self.kmeans.predict(X)
        X = pd.DataFrame(X)
        X['label'] = labels
        return X
    
    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init= self.n_init)
        self.kmeans.fit(X)
        return self



outliers_features= ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 
                    'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS',
                    'PRC_FULL_PAYMENT']

imputation_features = ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']