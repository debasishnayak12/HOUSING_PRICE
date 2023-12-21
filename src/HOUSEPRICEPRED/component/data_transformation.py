import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.HOUSEPRICEPRED.utils.utils import save_object
from src.HOUSEPRICEPRED.logger import logging
from src.HOUSEPRICEPRED.exception import customexception
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation(self):
        try:
            logging.info("data transformation started")
            
            num_columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
            cat_columns=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
            
            mainroad_cat=["yes","no"]
            guestroom_cat=["yes","no"]
            basement_cat=["yes","no"]
            hotwaterheating_cat=["yes","no"]
            airconditioning_cat=["yes","no"]
            prefarea_cat=["yes","no"]
            furnishingstatus_cat=['furnished', 'semi-furnished', 'unfurnished']
            
            logging.info("Pipeline initiated")
            
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer()),
                    ("Scaler", StandardScaler())
                ]
            )

            
            cat_pipeline=Pipeline(
                steps=[("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("Encoder",OrdinalEncoder(categories=[mainroad_cat,guestroom_cat,basement_cat,hotwaterheating_cat,airconditioning_cat,prefarea_cat,furnishingstatus_cat])),
                    ("Scaler",StandardScaler())]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )
            
            return preprocessor
    
    
        except Exception as e:
            logging.info("Error occured in get_data_transformation")
            raise customexception(e,sys)
        
    def initialize_data_transforamtion(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read tran and test data completed ")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessor_obj=self.get_data_transformation()
            
            target_column_name="price"
            
            input_feature_train_df=train_df.drop(labels=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(labels=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(f"shape of input_feature_train_df :{input_feature_train_df.shape}")
            logging.info(f"shape of input_feature_test_df :{input_feature_test_df.shape}")
            #X_train,X_test,y_train,y_test=train_test_split(input_feature_test_df,y,test_size=0.30,random_state=40)
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            logging.info("applying preprocessor_obj to train and test data")
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Preprocessor.pkl file saved")

            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Error occured in initialize dta transformation")
            raise customexception(e,sys)