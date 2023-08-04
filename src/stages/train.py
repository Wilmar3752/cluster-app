import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from utils import ScalerDf, Kmeans_, logger, load_config, outliers_features, imputation_features
import joblib


def train(data, config):
    logger.info('Training pipeline')
    pipeline_steps = [
        ('mean_inputer', MeanMedianImputer(imputation_method=config['imputation_method'], variables=imputation_features)),
        ('outlier_handling', Winsorizer(capping_method=config['capping_method'], variables=outliers_features, tail=config['tail'], fold=config['fold'])),
        ('standard', ScalerDf(method=config['scaler_method'])),
        ('PCA', PCA(n_components=config['n_components'])),
        ('Kmeans',Kmeans_(n_clusters=config['n_clusters']))
    ]

    cluster_pipeline = Pipeline(pipeline_steps)
    cluster_pipeline.fit(cc_data)
    logger.info('Export Cluster Pipeline')
    joblib.dump(cluster_pipeline, config['PIPELINE_PATH'])

if __name__ == "__main__":
    config = load_config('./', 'params.yaml')
    cc_data = pd.read_csv(config['DATA_PATH'], index_col=config['INDEX_COL'])
    train(cc_data, config)