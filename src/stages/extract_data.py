import logging
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
import argparse
from src.utils import load_config
import os

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def extract_data(config_path):
    config = load_config(config_path)
    username = config['data_load']['KAGGLE_USERNAME']
    dataset_name = config['data_load']['KAGGLE_DATASET_NAME']
    logger.info('Connection to Kaggle API')
    api = KaggleApi()
    api.authenticate()
    logger.info(f'Downloading data from Kaggle {username}-{dataset_name}')
    api.dataset_download_files(f'{username}/{dataset_name}', path=config['data_load']['PATH'], unzip=True)
    full_path = config['data_load']['PATH'] + '/CC GENERAL.csv'
    os.rename(full_path, config['data_load']['PATH'] +'/raw_data.csv')



if __name__=='__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    extract_data(args.config)