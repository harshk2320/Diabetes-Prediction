import json
import sys
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys) from e
        

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is requirred column present: [{status}]")
            # print("dataframe_cols: ", dataframe.columns)
            # print("_schema_cols: ",self._schema_config["columns"])
            return status
        except Exception as e:
            raise MyException(e, sys)
        

    def is_column_exit(self, df: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing Numerical Columns: {missing_numerical_columns}")

            return False if len(missing_numerical_columns) > 0 else True
        
        except Exception as e:
            raise MyException(e, sys) from e  
            
    
    @staticmethod
    def read_data(file_path):
        """
        This method reads data from CSV file and loads it as DataFrame.
        """
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise MyException(e, sys) from e
    
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        This method is responsible for validating training and testing datasets
        before they are used in training_pipeline
        
        1.  Ensure expected columns exist.
        2.  The data types of columns match the expectation.
        3.  Validation report is generated and stored as JSON file.
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path= self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path= self.data_ingestion_artifact.test_file_path ))
            # read() method reads data from CSV and loads it as DataFrame


            # Validating the no of columns
            status = self.validate_number_of_columns(train_df)
            if not status:
                validation_error_msg += f"columns are missing in train dataframe: {train_df.columns}"
            else:
                logging.info("All required columns are present in the training dataset: {status}")

            status = self.validate_number_of_columns(test_df)
            if not status:
                validation_error_msg += f"columns are missing in test dataset: {test_df.columns}"
            else:
                logging.info("All required columns are present in the test dataset: {status}")

            
            # Validating if the requirred columns are present are not.
            status = self.is_column_exit(df= train_df)
            if not status:
                validation_error_msg += f"columns are missing in train dataset."
            else:
                logging.info(f"All the required columns are present in the training dataset: {status}")

            status = self.is_column_exit(df= test_df)
            if not status:
                validation_error_msg += f"columns are missing in the test dataset."
            else:
                logging.info(f"All the required columns are present in the test dataset.")

            validation_status = len(validation_error_msg) == 0
            data_validation_artifact = DataValidationArtifact(
                                        validation_status= validation_status,
                                        message= validation_error_msg,
                                        validation_report_file_path= self.data_validation_config.validation_report_file_path)
            
            # Ensure the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok= True)

            # Saving validation status and message in to JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent= 4)

            logging.info("Data Validation artifacts created and saved in JSON file.")
            logging.info(f"Data Validation Artifact:{data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
        