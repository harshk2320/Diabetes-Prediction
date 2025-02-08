import sys
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ClassificationMetricArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                        model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model
        """
        self.data_transformation_artifact= data_transformation_artifact
        self.model_trainer_config= model_trainer_config


    def get_model_object_and_report(self, train: np.array, test:np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Training LogisticRegression with specified parametres.")
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")
            model = LinearRegression(
                fit_intercept= self.model_trainer_config._fit_intercept)
            
            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(f1_score= f1, precision_score= precision,
                                                           recall_score= recall, accuracy= accuracy)
            
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
        
            
    