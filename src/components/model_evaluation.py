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
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator
from sklearn.metrics import f1_score
import sys
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
    is_mode_accepted: bool
    difference: float

class ModelEvaluation:
    