import pandas as pd
import numpy as np
from src.HOUSEPRICEPRED.logger import logging
from src.HOUSEPRICEPRED.exception import customexception
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataingestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw_data.csv")
    train_data_path:str=os.path.join("artifacts","train_data.csv")
    test_data_path:str=os.path.join("artifacts","test_data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataingestionConfig()
        
        
    def initiate_dataingestion(self):
        logging.info("Dataingestion started")
        
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","Housing.csv")))
            logging.info("i have read dataset as data")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("i have saved the data as raw_data.csv in artifacts folder")
            
            train_data,test_data=train_test_split(data,test_size=0.3)
            logging.info("i have performed train_test_split")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("i have saved trainand test data in artifact as train_data and test_data")
            logging.info("Dataingestion part completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )    
        except Exception as e:
            logging.info("Error occured in Dataingestion")
            raise customexception(e,sys)
        

