import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from src.utils import ScalerDf, Kmeans_, logger, load_config, outliers_features, imputation_features
import joblib
import argparse
from pathlib import Path


def train(config_path):
    config = load_config(config_path)
    cc_data = pd.read_csv(Path(config['data_load']['PATH']) / config['data_load']['dataset_rename'], index_col=config['INDEX_COL'])
    print(cc_data)
    logger.info('Training pipeline')
    pipeline_steps = [
        ('mean_inputer', MeanMedianImputer(imputation_method=config['train']['imputation_method'], variables=imputation_features)),
        ('outlier_handling', Winsorizer(capping_method=config['train']['capping_method'], variables=outliers_features, tail=config['train']['tail'], fold=config['train']['fold'])),
        ('standard', ScalerDf(method=config['train']['scaler_method'])),
        ('PCA', PCA(n_components=config['train']['n_components'])),
        ('Kmeans',Kmeans_(n_clusters=config['train']['n_clusters']))
    ]

    cluster_pipeline = Pipeline(pipeline_steps)
    cluster_pipeline.fit(cc_data)
    logger.info('Export Cluster Pipeline')
    joblib.dump(cluster_pipeline, config['PIPELINE_PATH'])
    transformed_data = cluster_pipeline.transform(cc_data)
    transformed_data.to_csv(config['train']['transformed_data'], index=False)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train(args.config)