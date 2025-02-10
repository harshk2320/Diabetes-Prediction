"""
Purpose:
This code defines a ModelPusher class that is responsible for pushing (uploading) a
trained model to a remote storage system (like an S3 bucket) after it has passed evaluation. 
This is typically the final step in a machine learning pipeline where the best performing model is
saved to cloud storage for deployment or further use.
"""



import sys

from src.logger import logging
from src.exception import MyException
from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import ModelPusherArtifacts, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact, model_pusher_config: ModelPusherConfig ):
        """
        model_evaluation_artifact: Contains the results of model evaluation (like the path to the trained model and performance metrics).
        model_pusher_config: Configuration settings related to the model pushing process, including the bucket name and model path.
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(bucket_name= model_pusher_config.bucket_name,
                                               model_path= model_pusher_config.s3_model_key_path)
        