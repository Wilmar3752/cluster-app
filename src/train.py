import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.imputation import MeanMedianImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from utils import ScalerDf, Kmeans_, logger, load_config
import joblib


def train(data, config):
    logger.info('Training pipeline')
    pipeline_steps = [
        ('mean_inputer', MeanMedianImputer(imputation_method=config['imputation_method'], variables= ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT'])),
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