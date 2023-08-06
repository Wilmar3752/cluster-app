from sklearn.metrics import silhouette_score
from typing import Text
import pandas as pd
from src.utils import load_config
import logging, sys
from pathlib import Path
import matplotlib.pyplot as plt
import json
import argparse

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger('Evaluate')

def plot_clustering(transformed_data):
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.scatter(transformed_data['0'], transformed_data['1'], c=transformed_data['label'])

    ax.set_title('Datas with clusters', fontsize=16)

    ax.set_xlabel('Component 1', fontsize=13)
    ax.set_ylabel('Component 2', fontsize=13)
    return plt.gcf()

def evaluate_cluster(config_path: Text):
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """
    config = load_config(config_path)
    logger.info('Loading Transformed Data')

    transformed_data = pd.read_csv(config['train']['transformed_data'], index_col=False)
    print(transformed_data.head())
    plt = plot_clustering(transformed_data)
    plt.savefig(config['evaluate']['cluster_plot'])

    silhouette = silhouette_score(transformed_data.drop(columns='label'), transformed_data['label'])

    json.dump(
        obj={
            'Silhouette_score':silhouette
        },
    fp=open(config['evaluate']['metrics'], 'w')
    )
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_cluster(config_path=args.config)