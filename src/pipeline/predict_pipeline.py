import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os



class PredictPipeline:
    def __init__(self):
        pass

    
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
               Cough_symptoms: bool,
               Fever: bool,
               Sore_throat: bool,
               Shortness_of_breath: bool,
               Headache: bool,
               Age_60_above: str,
               Sex: str,
               Known_contact: str,
                 ):
        
        self.Cough_symptoms = Cough_symptoms
        self.Fever = Fever
        self.Sore_throat = Sore_throat
        self.Shortness_of_breath = Shortness_of_breath
        self.Headache = Headache
        self.Age_60_above = Age_60_above
        self.Sex = Sex
        self.Known_contact = Known_contact

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Cough_symptoms": [self.Cough_symptoms],
                "Fever": [self.Fever],
                "Sore_throat": [self.Sore_throat],
                "Shortness_of_breath": [self.Shortness_of_breath],
                "Headache": [self.Headache],
                "Age_60_above": [self.Age_60_above],
                "Sex": [self.Sex],
                "Known_contact": [self.Known_contact],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)