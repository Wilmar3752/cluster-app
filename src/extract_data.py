import logging
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from utils import logger

def extract_data(username, dataset_name):
    logger.info('Connection to Kaggle API')
    api = KaggleApi()
    api.authenticate()
    logger.info(f'Downloading data from Kaggle {username}-{dataset_name}')
    api.dataset_download_files(f'{username}/{dataset_name}', path='data/raw', unzip=True)

if __name__=='__main__':
    extract_data('arjunbhasin2013','ccdata')