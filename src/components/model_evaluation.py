"""
This code snippet is part of a model evaluation process where a trained machine
learning model is evaluated against a model deployed in a production environment.
It focuses on comparing the newly trained model’s performance with the current production
model and deciding whether the newly trained model should be accepted based on its
performance (in this case, F1 score).
"""
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.exception import MyException
from src.logger import logging
from src.constants import TARGET_COLUMN, DEPENDENT_COLUMNS
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator
from sklearn.metrics import f1_score
from sklearn.impute import KNNImputer
import sys
import numpy as np
from typing import Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class EvaluateModelResponse:
    """
    This is a dataclass used to encapsulate the evaluation results of the models. It contains:
    trained_model_f1_score: F1 score of the newly trained model.
    best_model_f1_score: F1 score of the best model from production (if available).
    is_model_accepted: A boolean indicating whether the newly trained model should be accepted based on its F1 score compared to the production model.
    difference: The difference in F1 score between the trained model and the production model.    
    """
    trainer_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact
                        , model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        This method is used to check if the production model exists and retrieve it.
        It uses a custom Proj1Estimator class to access the model from an S3 bucket. If the model exists, it returns it; otherwise, it returns None.
        If the model isn't found in the production environment, it returns None.
        """
        try:
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name= bucket_name, model_path= model_path)
            if proj1_estimator.is_model_present(model_path= model_path):
                return proj1_estimator
            return None
        
        except Exception as e:
            raise MyException(e, sys) from e
        
    def drop_id_column(self, df):
        """
        Dropping 'id' column if it exists
        """
        logging.info("Dropping id column.")
        
        if "_id" in df.columns:
            df = df.drop("_id", axis = 1)
        
        return df
    

    def handling_noise(self, df):
        """
        Removing/handling noise values by replacing them with nulls
        """
        logging.info("Removing noise values.")
        for i in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]:
            df.loc[df[i] == 0, i] = np.nan        
        
        return df


    def handling_nulls(self, df):
        """
        Removing null values using KNN
        """
        logging.info("Removing null values.")
        knn_inputer = KNNImputer(n_neighbors= 6, weights= "distance")
        df = pd.DataFrame(knn_inputer.fit_transform(df), columns= df.columns)
        print(df.isna().sum())
        return df


    def handling_outliers(self, df):
        """
        Removing outliers using IQR method.
        """
        logging.info("Removing outliers using IQR method.")
        for i in DEPENDENT_COLUMNS:
            q1 = df[i].quantile(0.25)
            q3 = df[i].quantile(0.75)
            iqr = q3 - q1
            lc = q1 - 1.5 * iqr
            uc = q3 + 1.5 * iqr

            df[i] = df[i].clip(lower= lc, upper= uc)
        
        return df
    

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Main goal: Compares the newly trained model with the best model from production (if it exists)
        and decides whether the newly trained model should be accepted.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis= 1), test_df[TARGET_COLUMN]
            logging.info("Test data loaded and now transforming it for predictions...")
            
            x = self.drop_id_column(x)
            x = self.handling_noise(x)
            x = self.handling_nulls(x)
            x = self.handling_outliers(x)

            trained_model = load_object(file_path= self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"f1_score for this model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Computing f1_score for production model...")
                y_hat_best_model = best_model.predict()
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"f1_score for production model: {best_model_f1_score}, f1_score for newly trained model: {trained_model_f1_score}")

            temp_best_model_f1_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trainer_model_f1_score= trained_model_f1_score,
                                           best_model_f1_score= best_model_f1_score,
                                           is_model_accepted= trained_model_f1_score > temp_best_model_f1_score,
                                           difference= trained_model_f1_score - temp_best_model_f1_score)
            
            logging.info(f"Result: {result}")
            return result
        
        except Exception as e:
            raise MyException(e, sys) from e
        
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("--------------------------------------------------------------------------------------------")
            logging.info("Initialized model evaluation component.")
            evaluate_model_response = self.evaluate_model()  
            s3_model_path = self.model_evaluation_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted= evaluate_model_response.is_model_accepted,
                changed_accuracy= evaluate_model_response.difference,
                s3_model_path= s3_model_path,
                trained_model_path= self.model_trainer_artifact.trained_model_file_path)
            
            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        