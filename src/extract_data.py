import logging
import sys
from kaggle.api.kaggle_api_extended import KaggleApi


    # Setup logging configuration
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def extract_data(username, dataset_name):
    logger.info('Connection to Kaggle API')
    api = KaggleApi()
    api.authenticate()
    logger.info(f'Downloading data from Kaggle {username}-{dataset_name}')
    api.dataset_download_files(f'{username}/{dataset_name}', path='data/raw', unzip=True)

if __name__=='__main__':
    extract_data('arjunbhasin2013','ccdata')