import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.HOUSEPRICEPRED.logger import logging 
from src.HOUSEPRICEPRED.exception import customexception
from src.HOUSEPRICEPRED.utils.utils import save_object
from src.HOUSEPRICEPRED.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("splitting dependent and independent data from train and test data")
            
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n===================================================================\n")
            logging.info(f"model report : {model_report}")
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            print(f"Best model found : model name ={best_model_name} ,R2_score = {best_model_score}")
            print("\n=====================================================================\n")
            logging.info(f"Best model found : model name ={best_model_name} ,R2_score = {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            
        except Exception as e:
            logging.info("Error occured in initiate model training model trainer file")
            raise customexception(e,sys)
    