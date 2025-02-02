import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.Data_transformation_Config = DataTransformationConfig()

    def get_data_transformation_obj(self):

        '''
        This Function is responsible to process
        and transform data for modeling
        '''

        try:
            categorical_features = ['education', 'self_employed']
            numeric_features = ['no_of_dependents', 'income_annum', 'loan_amount',
                                'loan_term', 'income_to_loan_ratio', 'loan_to_cibil_ratio',
                                'loan_term_to_income_ratio', 'assets']
            num_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='median')),
                    ('Scaler', MinMaxScaler())
                ]
            )
            logging.info('Numerical Scaling Completed')
            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHot', OneHotEncoder(drop='first'))
                ]
            )

            logging.info('Categorical Encoding Completed')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('Numerical-Pipeline', num_pipeline, numeric_features),
                    ('Categorical-Pipeline', cat_pipeline, categorical_features),
                ])

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test sets Completed')
            train_df.columns = train_df.columns.str.strip().str.replace(' ', '')
            test_df.columns = test_df.columns.str.strip().str.replace(' ', '')

            for features in ['education', 'self_employed', 'loan_status']:
                train_df[features] = train_df[features].apply(lambda x: x.strip())
                test_df[features] = test_df[features].apply(lambda x: x.strip())

            train_df["income_to_loan_ratio"] = train_df["income_annum"] / train_df["loan_amount"]
            train_df["loan_to_cibil_ratio"] = train_df["loan_amount"] / train_df["cibil_score"]
            train_df["loan_term_to_income_ratio"] = train_df['loan_term'] / train_df["income_annum"]

            test_df["income_to_loan_ratio"] = test_df["income_annum"] / test_df["loan_amount"]
            test_df["loan_to_cibil_ratio"] = test_df["loan_amount"] / test_df["cibil_score"]
            test_df["loan_term_to_income_ratio"] = test_df["loan_term"] / test_df["income_annum"]
            logging.info('Derived columns added')
            train_df['assets'] = train_df['residential_assets_value'] + train_df['commercial_assets_value'] + train_df[
                'luxury_assets_value'] + train_df['bank_asset_value']
            test_df['assets'] = test_df['residential_assets_value'] + test_df['commercial_assets_value'] + test_df[
                'luxury_assets_value'] + test_df['bank_asset_value']

            train_df.drop(columns=['loan_id', 'residential_assets_value', 'commercial_assets_value',
                                   'luxury_assets_value', 'bank_asset_value','cibil_score'], inplace=True)
            test_df.drop(columns=['loan_id', 'residential_assets_value', 'commercial_assets_value',
                                  'luxury_assets_value', 'bank_asset_value', 'cibil_score'], inplace=True)

            logging.info('Derived columns and asset calculation completed')

            label_encoder = LabelEncoder()
            train_df['loan_status'] = label_encoder.fit_transform(train_df['loan_status'])
            test_df['loan_status'] = label_encoder.transform(test_df['loan_status'])

            logging.info('Label Encoding applied to loan_status column')
            logging.info('Obtaining Preprocessing object')

            preprocessor_obj = self.get_data_transformation_obj()

            categorical_features = ['education', 'self_employed']
            numeric_features = ['no_of_dependents', "income_annum", "loan_amount", "loan_term", "assets",
                                "income_to_loan_ratio", "loan_to_cibil_ratio", "loan_term_to_income_ratio"]

            input_feature_train_df = train_df.drop(columns=["loan_status"])
            target_feature_train_df = train_df["loan_status"]
            input_feature_test_df = test_df.drop(columns=["loan_status"])
            target_feature_test_df = test_df["loan_status"]

            logging.info('Applying preprocessing on training dataframe and testing dataframe')
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved processing object')

            save_obj(
                file_path=self.Data_transformation_Config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.Data_transformation_Config.preprocessor_ob_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
