import sys
import yaml
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import save_obj, evaluate_model
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting Training and Test Input Data')
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            logging.info(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

            models = {
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'RandomForest': RandomForestClassifier(random_state=42),
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'ExtraTreeClassifier': ExtraTreesClassifier(random_state=42),
                'KNN': KNeighborsClassifier(),
                'BaggingClassifier': BaggingClassifier(estimator=RandomForestClassifier(random_state=42)),
                'GradientBoost': GradientBoostingClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42)
            }

            parameters = {
                'LogisticRegression': {
                    'C': [0.01,0.1,1],
                    'penalty': ['l2']
                },
                'SVM': {
                    'C': [0.01, 0.1, 1],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                },

                'DecisionTree': {
                    'criterion': ['gini'],
                    'max_depth': [1, 2, 3, 4],
                    'min_samples_split': [10, 20, 30],
                    'min_samples_leaf': [5, 10, 15],
                    'max_features': ['sqrt', 'log2']
                },

                'ExtraTreeClassifier': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10, 15],  # Restrict tree depth to prevent overfitting
                },

                'KNN': {
                    'n_neighbors': [5, 10, 15, 20],
                    'weights': ['uniform', 'distance'],
                },

                'BaggingClassifier': {
                    'n_estimators': [50, 100, 150],
                    'max_samples': [0.5, 0.7],
                    'max_features': [0.5, 0.7, 1.0],
                    'bootstrap': [True]
                }

                ,

                'GradientBoost': {
                    'n_estimators': [50, 100, 200],  # Increased for better learning
                    'learning_rate': [0.01, 0.05, 0.1],  # Kept the same
                    'max_depth': [3, 5, 7, 10],  # Reduced to prevent overfitting
                    'min_samples_split': [5, 10, 20],  # More flexibility in splits
                    'min_samples_leaf': [1, 3, 5, 10],  # Allows learning small patterns
                    'subsample': [0.8, 0.9],  # Removed 1.0 to encourage randomness
                    'validation_fraction': [0.1, 0.2],  # Kept reasonable
                    'n_iter_no_change': [5, 10, 20],  # No need for 30
                    'ccp_alpha': [0.0, 0.001, 0.01]  # Finer regularization control
},

                'AdaBoost': {
                    'n_estimators': [50, 100, 150],  # More weak learners
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Better granularity
}
            }

            model_report, best_model_name = evaluate_model(X_train, X_test, y_train, y_test, models,parameters)

            # Log model metrics
            logging.info("All Models' Metrics:")
            for model_name, metrics in model_report.items():
                logging.info(f"Model: {model_name}")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value}")
                logging.info("-" * 60)

            # Log and save best model based on Test F1-Score
            logging.info(
                f"Best Model: {best_model_name} with Test F1-Score: {model_report[best_model_name]['test_f1']}")
            best_model = models[best_model_name]
            logging.info("Model Training Completed")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)
