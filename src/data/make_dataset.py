import os
import sys
import zipfile
from src.logger import logging
from src.exception import CustomException
from pathlib import Path


def extract(intput_path:Path,output_path:Path):
    with zipfile.ZipFile( intput_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)



if __name__=='__main__':
    try:
        input_path=Path('data/raw/Celeb.zip')
        output_path=Path('data/processed')
        extract(input_path,output_path)
        logging.info("Data is extracted sucessfully")
    except Exception as e:
        raise CustomException(e,sys)
    
