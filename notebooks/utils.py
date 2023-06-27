from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

class ScalerDf(BaseEstimator, TransformerMixin):

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
        