from dataclasses import dataclass       # Provides @dataclass decorator to define configuration class

@dataclass      # It is a python decorator which automatically generates method (__init__, __rep__).
                # Used for storing configuration without manually writing code  
class DataIngestionArtifact:
    trained_file_path:str      # Path where the trained data is stored
    test_file_path:str         # Path where test data is stored


@dataclass      # Its is a python decorator which automatically takes method (__init__, __rep__ etc)
                # Used for storing configurations without manually writing boiler code

class DataValidationArtifact:
    validation_status:bool      
    message:str
    validation_report_file_path:str

@dataclass      # Its is a python decorator which automatically takes method (__init__, __rep__ etc)
                # Its is used for storing configurations without writing boiler codes
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str



# @dataclass
# class DataValidationArtifact:
#     validation_status:bool
#     message: str
#     validation_report_file_path: str

# @dataclass
# class DataTransformationArtifact:
#     transformed_object_file_path:str 
#     transformed_train_file_path:str
#     transformed_test_file_path:str

# @dataclass
# class ClassificationMetricArtifact:
#     f1_score:float
#     precision_score:float
#     recall_score:float

# @dataclass
# class ModelTrainerArtifact:
#     trained_model_file_path:str 
#     metric_artifact:ClassificationMetricArtifact

# @dataclass
# class ModelEvaluationArtifact:
#     is_model_accepted:bool
#     changed_accuracy:float
#     s3_model_path:str 
#     trained_model_path:str

# @dataclass
# class ModelPusherArtifact:
#     bucket_name:str
#     s3_model_path:str