from src.HOUSEPRICEPRED.component.data_ingestion import DataIngestion
from src.HOUSEPRICEPRED.component.data_transformation import DataTransformation
from src.HOUSEPRICEPRED.component.model_trainer import ModelTrainer

import os
import sys
import pandas as pd
from src.HOUSEPRICEPRED.exception import customexception
from src.HOUSEPRICEPRED.logger import logging

obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_dataingestion()

data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.initialize_data_transforamtion(train_data_path,test_data_path)

model_trainer=ModelTrainer()
model_trainer.initiate_model_training(train_arr,test_arr)



