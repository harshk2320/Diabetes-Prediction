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
        