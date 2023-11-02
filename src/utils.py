import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV




from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            # Extract the hyperparameter grid for the current model
            hyperparameter_grid = param.get(model_name, {})

            # If hyperparameter grid is provided, perform GridSearchCV
            if hyperparameter_grid:
                gs = GridSearchCV(model, hyperparameter_grid, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            # Train the model
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            train_precision = precision_score(y_train, y_train_pred)
            test_precision = precision_score(y_test, y_test_pred)
            
            train_recall = recall_score(y_train, y_train_pred)
            test_recall = recall_score(y_test, y_test_pred)
            
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            report[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'train_f1_score': train_f1,
                'test_f1_score': test_f1,
            }

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)