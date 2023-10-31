import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            columns = [
                "Cough_symptoms",
                "Fever",
                "Sore_throat",
                "Shortness_of_breath",
                "Headache",
                "Age_60_above",
                "Sex",
                "Known_contact",
            ]

            pipeline=Pipeline(
                steps=[
                ("one_hot_encoder",OneHotEncoder(drop='first'))
                ]

            )

            logging.info(f"Categorical columns: {columns}")

            preprocessor=ColumnTransformer(
                [
                ("pipeline",pipeline,columns)

                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Corona"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            input_feature_train_dense = input_feature_train_arr.toarray()
            input_feature_test_dense = input_feature_test_arr.toarray()

            # column_names=["Cough_symptoms","Fever","Sore_throat","Shortness_of_breath","Headache","Age_60_above","Sex","Known_contact"]

            input_feature_train = pd.DataFrame(input_feature_train_dense)
            input_feature_test = pd.DataFrame(input_feature_test_dense)
            print(input_feature_train)
            print(target_feature_train_df)

            train_arr = pd.concat([input_feature_train, target_feature_train_df], axis=1)
            test_arr = pd.concat([input_feature_test, target_feature_test_df], axis=1)

            # train_arr = np.column_stack((input_feature_train_arr, np.array(target_feature_train_df)))
            # test_arr = np.column_stack((input_feature_test_arr, np.array(target_feature_test_df)))

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )




        except Exception as e:
            raise CustomException(e,sys)