import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

from src.constants import CURRENT_YEAR, TARGET_COLUMN, SCHEMA_FILE_PATH, DEPENDENT_COLUMNS
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTranformation:
    def __init__(self, data_ingestion_artifact = DataIngestionArtifact,
                data_transformation_config= DataTransformationConfig,
                data_validation_artifact = DataValidationArtifact):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path= SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    
    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(sys, e)

    def get_data_transformer_object(self):
        """
        This method is responsible for creating and returning a preprocessing pipeline for transforming 
        the dataset before feeding it to the ML model.
        """

        logging.info("Entered the get_data_transformation_object of DataTransformation Class.")

        try:
            # Initializing transformers
            numeric_scaler = StandardScaler()
            logging.info("Transformer Initialized: StandardScaler")

            num_features = self._schema_config["num_features"]
            logging.info("Columns loaded from schema")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(transformers= ("StandardScaler", numeric_scaler),
                                             remainder= "passthrough")
            
            # Wrapping everything in a single Pipeline
            final_pipeline = Pipeline(steps= ("Preprocessor", preprocessor))
            logging.info("Final pipeline ready!!")
            logging.info("Exited get_data_transformation_object of DataTransformation class.")
            return final_pipeline
        
        except Exception as e:
            logging.exception("Exception occured in get_data_transformation_object of DataTransformation Class.")
            raise MyException(e, sys)
        
        
    def drop_id_column(self, df):
        """
        Dropping 'id' column if it exists
        """
        logging.info("Dropping id column.")
        drop_col = self._schema_config["drop_columns"]
        if drop_col in df.columns:
            df = df.drop(columns= drop_col)
        
        return df
    

    def handling_noise(self, df):
        """
        Removing/handling noise values by replacing them with nulls
        """
        logging.info("Removing noise values.")
        for i in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]:
            df.loc[df[i] == 0, i] = np.nan        
        
        return df


    def handling_nulls(self, df):
        """
        Removing null values using KNN
        """
        logging.info("Removing null values.")
        knn_inputer = KNNImputer(n_neighbors= 6, weights= "distance")
        df = pd.DataFrame(knn_inputer.fit_transform(df), columns= df.columns)

        return df


    def handling_outliers(self, df):
        """
        Removing outliers using IQR method.
        """
        logging.info("Removing outliers using IQR method.")
        for i in DEPENDENT_COLUMNS:
            q1 = df[i].quantile(0.25)
            q3 = df[i].quantile(0.75)
            iqr = q3 - q1
            lc = q1 - 1.5 * iqr
            uc = q3 + 1.5 * iqr

            df[i] = df[i].clip(lower= lc, upper= uc)
        
        return df
    

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component of the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df, test_df = (self.read_data(self.data_ingestion_artifact.trained_file_path),
                                self.read_data(self.data_ingestion_artifact.test_file_path))
            logging.info("Train-Test data loaded.")

            input_feature_train_df = train_df.drop(columns= TARGET_COLUMN)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns= TARGET_COLUMN)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Input and Target columns defined for both train & test df.")

            # Apply custom transformation in the specified sequence
            input_feature_train_df = self.drop_id_column(input_feature_train_df)
            input_feature_test_df = self.handling_noise(input_feature_train_df)            
            input_feature_test_df = self.handling_nulls(input_feature_train_df)
            input_feature_test_df = self.handling_outliers(input_feature_train_df)

            input_feature_test_df = self.drop_id_column(input_feature_test_df)        
            input_feature_test_df = self.handling_noise(input_feature_test_df)
            input_feature_test_df = self.handling_nulls(input_feature_test_df)
            input_feature_test_df = self.handling_outliers(input_feature_test_df)
            logging.info("Custom Transformation applied to both train and test data.")

            logging.info("Starting data transformation")
            preprocesssor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training data.")
            input_feature_train_arr = preprocesssor.fit_transform(input_feature_train_df)

            logging.info("Initializing transformation for Testing data.")
            input_feature_test_arr = preprocesssor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test data.")

            logging.info("Applying SMOTEENN for handling imbalance dataset.")
            smt = SMOTEENN(sampling_strategy= "minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr,
                                                                                     target_feature_train_df)
            
            input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, 
                                                                                   target_feature_test_df)

            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df")

            save_object(self.data_transformation_config.tranformed_object_file_path, preprocesssor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation applied successfully")
            return DataValidationArtifact(
                transformed_object_file_path = self.data_transformation_config.tranformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path)
            
        except Exception as e:
            raise MyException(e, sys) from e
        













    
