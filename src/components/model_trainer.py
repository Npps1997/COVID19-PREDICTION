import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array.iloc[:,:-1],
                train_array.iloc[:,-1],
                test_array.iloc[:,:-1],
                test_array.iloc[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "GaussianNB": GaussianNB(),
                "KNN": KNeighborsClassifier(),
            }

            params = {
                        "Random Forest": {
                            'n_estimators': [8, 16, 32, 64],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None, 10, 20, 30, 40, 50],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'max_features': ['auto', 'sqrt', 'log2'],
                        },
                        "Logistic Regression": {
                            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                            'C': [0.001, 0.01, 0.1, 1, 10],
                            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                            'max_iter': [100, 1000, 10000],
                        },
                        "CatBoosting Classifier": {
                            'depth': [6, 8, 10],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'iterations': [30, 50, 100],
                        },
                        "AdaBoost Classifier": {
                            'n_estimators': [8, 16, 32, 64, 128, 256],
                            'learning_rate': [0.1, 0.01, 0.001],
                            # 'algorithm': ['SAMME', 'SAMME.R'],
                        },
                        "GaussianNB": {},
                        "KNN": {
                            'n_neighbors': [3, 5, 7, 9],
                            'weights': ['uniform', 'distance'],
                            'p': [1, 2],
                        },
                    }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, param=params)


            ## To get best model score from dict
            best_model_score = max(model_report.values(), key=lambda x: x['test_accuracy'])['test_accuracy']

            ## To get best model name from dict

            best_model_name = [model for model, scores in model_report.items() if scores['test_accuracy'] == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            Accuracy_score = accuracy_score(y_test, predicted)
            return Accuracy_score

        except Exception as e:
            raise CustomException(e,sys)