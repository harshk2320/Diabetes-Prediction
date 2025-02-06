import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.constants import CURRENT_YEAR, TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTranformation:
    def __init__(self, data_ingestion_artifact = DataIngestionArtifact,
                data_transformation_config= DataTransformationConfig,
                data_validation_artifact = DataValidationArtifact):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path= SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    
    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(sys, e)

    def get_data_transformer_object(self):
        """
        This method is responsible for creating and returning a preprocessing pipeline for transforming 
        the dataset before feeding it to the ML model.
        """

        logging.info("Entered the get_data_transformation_object of DataTransformation Class.")

        try:
            # Initializing transformers
            numeric_scaler = StandardScaler()
            logging.info("Transformer Initialized: StandardScaler")

            num_features = self._schema_config["num_features"]
            logging.info("Columns loaded from schema")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(transformers= ("StandardScaler", numeric_scaler),
                                             remainder= "passthrough")
            
            # Wrapping everything in a single Pipeline
            final_pipeline = Pipeline(steps= ("Preprocessor", preprocessor))
            logging.info("Final pipeline ready!!")
            logging.info("Exited get_data_transformation_object of DataTransformation class.")
            return final_pipeline
        
        except Exception as e:
            logging.exception("Exception occured in get_data_transformation_object of DataTransformation Class.")
            raise MyException(e, sys)
        






















    
