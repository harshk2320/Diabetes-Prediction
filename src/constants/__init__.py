import os
from datetime import datetime

# For MOngoDB connection
DATABASE_NAME= "Proj2"
COLLECTION_NAME = "Proj2-Data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = ""             # this format of writing is called type hinting (introduced in python 3.5). Its only for better readabilty
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME= "LR_BestModel_Diabetes.pkl"

DEPENDENT_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET_COLUMN = "Outcome"
CURRENT_YEAR =  datetime.today().year
PREPROCESSING_OBJECT_FILE_NAME= "preprocessing.pkl"

FILE_NAME= "data.csv"
TRAIN_FILE_NAME= "train.csv"
TEST_FILE_NAME = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME= "us-east-1"

"""
Data Ingestion related constant start with DATA_INGESTION  var name
"""
DATA_INGESTION_COLLECTION_NAME: str = "Proj2-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion" 
DATA_INGESTION_FEATURE_STORE_DIR: str= "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: str = 0.25

"""
Data Validation related constant start with DATA_VALIDATION var name
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME:str = "report.yaml"

"""
Data Transformation related constant start with DATA_TRANSFORMATION
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "transformed_object"

"""
Model Trainer related constant start with MODEL_TRAINER var name.
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str =  "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE:float = 0.6 
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH:str = os.path.join("config", "model.yaml") 
MODEL_TRAINER_FIT_INTERCEPT: bool = False

"""
Model Evaluation related constants 
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE : float = 0.02
MODEL_BUCKET_NAME = "my-model-mlopsproj-diabetes-harsh"
MODEL_PUSHER_S3_KEY = "model-registry"














































