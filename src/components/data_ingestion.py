"""
DataIngestion is class responsible for extracting data from MongoDB storing it into a CSV file and then splitting it into training
and testing. This is the first step in ML pipeline"""

import os       # Handles file path operations
import sys      # Handles system related exceptions

from pandas import DataFrame    # Handles dataframe related operations
from sklearn.model_selection import train_test_split    # Performs train-test split

from src.entity.config_entity import DataIngestionConfig      # Configuration for data ingestion
from src.entity.artifact_entity import DataIngestionArtifact    # Output artifacts (file paths)
from src.exception import MyException     # Custom exception handling
from src.logger import logging      # Logging for debugging and monitoring
from src.data_access.proj2_data import Proj2Data    # MongoDB data extraction

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        Stores configuration for data ingestion
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        

    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb and saves it as csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")    # Logging the process
            my_data = Proj2Data()   # Creating an object of Proj2Data (MongoDB Data handler)
            dataframe = my_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.collection_name)
            # export_collection_as_dataframe() is used to get data from MongoDB and then return it as dataframe.


            logging.info(f"Shape of dataframe: {dataframe.shape}")  # Logging the no of rows and columns
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path   # Define file path
            dir_path = os.path.dirname(feature_store_file_path)     # get the directory path
            os.makedirs(dir_path,exist_ok=True)     # Creating directory if it does not exist
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")   # logging the process
            dataframe.to_csv(feature_store_file_path,index=False,header=True)       # Saving the data into CSV file for later use
            return dataframe      # Returns dataframe so that it can be used for splitting into train and test.

        except Exception as e:
            raise MyException(e,sys)

    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")     # Logging the process

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)     # Ensures the directory exit before saving the files
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            # Creates a DataIngestionArtifact which stores train and test file path
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            # Logs and returns the artifacts so that later pipeline steps can use the data

        except Exception as e:
            raise MyException(e, sys) from e