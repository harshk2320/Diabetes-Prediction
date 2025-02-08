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

