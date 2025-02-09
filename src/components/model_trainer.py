import sys
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ClassificationMetricArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel
from src.utils.main_utils import *

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
            model = LogisticRegression(
                fit_intercept= self.model_trainer_config._fit_intercept)
            
            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            # y_pred_proba = model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class
            # y_pred = (y_pred_proba >= 0.5).astype(int)
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
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class.")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("--------------------------------------------------------------------------------")
            print("Starting Model Trainer Component.")

            # Load tranform train and test data
            train_arr = load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")

            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train= train_arr, test= test_arr)
            logging.info("Model object and artifact loaded.")

            # Load preprocessing object
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], (trained_model.predict(train_arr[:, :-1])).astype(int)) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with the score above the base score.")
                raise Exception("No model found with the score above the base score.")

            logging.info("Saving the new model as the performance is better.")
            my_model = MyModel(preprocessing_object= preprocessing_obj, trained_model_object= trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both processing and the trained model.")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path= self.model_trainer_config.trained_model_file_path,
                                metric_artifact= metric_artifact)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            print(metric_artifact)
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
            
