import os
import sys
import pandas as pd

from src.HOUSEPRICEPRED.logger import logging
from src.HOUSEPRICEPRED.exception import customexception
from src.HOUSEPRICEPRED.utils.utils import load_obj

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,feature):
        try:
            logging.info("prediction pipeline started")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_obj(preprocessor_path)
            model=load_obj(model_path)
            
            scaled_data=preprocessor.transform(feature)
            pred=model.predict(scaled_data)
            
            return pred
        except Exception as e:
            raise customexception(e,sys)
        

class customdata:
    def __init__(self,
                 area:float,
                 bedrooms:float,
                 bathrooms:float,
                 stories:float,
                 mainroad:str,
                 guestroom:str,
                 basement:str,
                 hotwaterheating:str,
                 airconditioning:str,
                 parking:float,
                 prefarea:str,
                 furnishingstatus:str):
        
        self.area=area
        self.bedrooms=bedrooms
        self.bathrooms=bathrooms
        self.stories=stories
        self.mainroad=mainroad
        self.guestroom=guestroom
        self.basement=basement
        self.hotwaterheating=hotwaterheating
        self.airconditioning=airconditioning
        self.parking=parking
        self.prefarea=prefarea
        self.furnishingstatus=furnishingstatus
        
        
    def get_data_as_dataframe(self):
        try:
            customdata_input_dict={
                "area":[self.area],
                "bedrooms":[self.bedrooms],
                "bathrooms":[self.bathrooms],
                "stories":[self.stories],
                "mainroad":[self.mainroad],
                "guestroom":[self.guestroom],
                "basement":[self.basement],
                "hotwaterheating":[self.hotwaterheating],
                "airconditioning":[self.airconditioning],
                "parking":[self.parking],
                "prefarea":[self.prefarea],
                "furnishingstatus":[self.furnishingstatus]
            }
            
            df=pd.DataFrame(customdata_input_dict)
            
            logging.info("Dataframe gathered")
            return df
        except Exception as e:
            logging.info("error occured in predictionpipeline")
            raise customexception(e,sys)
        