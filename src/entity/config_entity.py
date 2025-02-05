"""
This code defines configurations for training a machine learning pipeline and data ingestion, data validation etc.
It uses dataclasses for better structure and readability.
"""


import os       # Handles file path operations
from src.constants import *     # Contain global constants like file paths and collection names
from dataclasses import dataclass       # Provides @dataclass decorator to define configuration class
from datetime import datetime       # Generates a timestamp to create unique directories

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")       # Generates a unique timestamp based on current date and time.
                                                                    # Used to seperate different artifacts directories different runs
                                                                    
                                                                    # Example Output
                                                                    # If the code is executed on January 29, 2025, at 14:05:30,
                                                                    # TIMESTAMP = "01_29_2025_14_05_30"
                                                                    
                                                                    
@dataclass      # it is a python decorator which automatically generates methods (__init__, __rep__ etc). 
                # Used for storing configurations without manually writing boilerplate codes                                      

class TrainingPipelineConfig:
    """
    This class stores High level configurations for ML training Pipelines
    """
    pipeline_name: str = PIPELINE_NAME    # Name of the pipeline which should be string
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)       # Path where output artifacts (processed data, models etc ) will be stored.
    timestamp: str = TIMESTAMP      # Ensure every run gets a timestamp of that run


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()     # Creating an object for TrainingPipelineConfig class

@dataclass      # Its is a python decorator which automatically generate method (__init__, __rep__)
                # Used for storing configurations without manually writing boiler plate codes
class DataIngestionConfig:
    """
    The purpose of this is to store configuration related to the Data Ingestion phase of the ML Pipeline.
    It ensure that all file paths, directories and parametres are defined properly
    """
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME) 
    # root directory for data ingestion artifacts
    
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    # Location to save raw dataset fetched from MongoDB

    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    # Location for saving the training dataset after splitting.

    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    # Location for saving the testing dataset after splitting

    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    # Percentage of data allocated for testing

    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    # MongoDB collection containing the raw data


# @dataclass
# class DataValidationConfig:
#     data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
#     validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

# @dataclass
# class DataTransformationConfig:
#     data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
#     transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
#                                                     TRAIN_FILE_NAME.replace("csv", "npy"))
#     transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
#                                                    TEST_FILE_NAME.replace("csv", "npy"))
#     transformed_object_file_path: str = os.path.join(data_transformation_dir, 
#                                                      DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
#                                                      PREPROCSSING_OBJECT_FILE_NAME)
    
# @dataclass
# class ModelTrainerConfig:
#     model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
#     trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
#     expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
#     model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
#     _n_estimators = MODEL_TRAINER_N_ESTIMATORS
#     _min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT
#     _min_samples_leaf = MODEL_TRAINER_MIN_SAMPLES_LEAF
#     _max_depth = MIN_SAMPLES_SPLIT_MAX_DEPTH
#     _criterion = MIN_SAMPLES_SPLIT_CRITERION
#     _random_state = MIN_SAMPLES_SPLIT_RANDOM_STATE

# @dataclass
# class ModelEvaluationConfig:
#     changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
#     bucket_name: str = MODEL_BUCKET_NAME
#     s3_model_key_path: str = MODEL_FILE_NAME

# @dataclass
# class ModelPusherConfig:
#     bucket_name: str = MODEL_BUCKET_NAME
#     s3_model_key_path: str = MODEL_FILE_NAME

# @dataclass
# class VehiclePredictorConfig:
#     model_file_path: str = MODEL_FILE_NAME
#     model_bucket_name: str = MODEL_BUCKET_NAME