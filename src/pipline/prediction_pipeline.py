import sys
import pandas as pd
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from entity.s3_estimator import Proj1Estimator
from src.entity.config_entity import DiabetesPredictionConfig


class DiabetesData:

    def __init__(self, Pregnancies, Glucose, BloodPressure, SkinThickness,
                 Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome):
        """
        Purpose: This function assigns these feature values to corresponding attributes of the class instance.

        """
        try:
            self.Pregnancies = Pregnancies
            self.Glucose = Glucose
            self.BloodPressure = BloodPressure
            self.SkinThickness = SkinThickness
            self.Insulin = Insulin
            self.BMI = BMI
            self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
            self.Age = Age
            self.Outcome = Outcome
        
        
        except Exception as e:
            raise MyException(e, sys) from e
        

    def get_diabetes_data_as_dict(self):
        """
        Purpose: Converts the input feature values into a dictionary format. 
        """

        logging.info("Entered get_diabetes_data_as_dict method of DiabetesData class.")
        try:
            input_data = {
            "Pregnancies" : [self.Pregnancies],
            "Glucose" : [self.Glucose],
            "BloodPressure" : [self.BloodPressure],
            "SkinThickness" : [self.SkinThickness] ,
            "Insulin" : [self.Insulin],
            "BMI" : [self.BMI],
            "DiabetesPedigreeFunction" : [self.DiabetesPedigreeFunction], 
            "Age" : [self.Age],
            "Outcome" : [self.Outcome] 
            }

            logging.info("Created Diabetes data dict.")
            logging.info("Exited get_diabetes_data_as_dict of DiabetesData class.")
        
            return input_data
        
        except Exception as e:
            raise MyException(e, sys) from e
        
        
    def get_diabetes_input_data_frame(self) -> DataFrame:
        """
        This function convert the input_data from get_diabtes_as_dict to dataframe.
        """
        try:
            diabetes_input_dict = self.get_diabetes_data_as_dict()
            return pd.DataFrame(diabetes_input_dict)

        except Exception as e:
            raise MyException(e, sys) from e
        

class DiabetesDataClassifier:
    """
    The VehicleDataClassifier class is responsible for taking in the vehicle data (as a DataFrame)
    and making predictions using a trained machine learning model.
    """
    
    def __init__(self, prediction_pipeline_config: DiabetesPredictionConfig = DiabetesPredictionConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        
        except Exception as e:
            raise MyException(e, sys) from e
        
    def predict(self, dataframe) -> str:
        """
        Purpose: This method takes in a dataframe (likely a pandas DataFrame containing the vehicle data), 
        loads the trained model from the S3 bucket, and uses the model to make a prediction.

        It initializes a Proj1Estimator object, which likely loads a pre-trained machine learning model from the S3 bucket.
        It calls model.predict(dataframe) to make the prediction using the model and the input data.
        Finally, it returns the result of the prediction (likely a class label or a numerical prediction, depending on the model).
               
        """

        try:
            logging.info("Entered predict method of DiabetesDataClassifier class.")
            model = Proj1Estimator(bucket_name= self.prediction_pipeline_config.model_bucket_name,
                                   model_path= self.prediction_pipeline_config.model_file_path)
            
            result = model.predict(dataframe)
            return result
        
        except Exception as e:
            raise MyException(e, sys) from e
        