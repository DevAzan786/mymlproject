import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

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

            models = {
                'LogisticRegression': LogisticRegression(),
                'SVM': SVC(),
                'RandomForest': RandomForestClassifier(),
                'DecisionTree': DecisionTreeClassifier(),
                'ExtraTreeClassifier': ExtraTreesClassifier(),
                'KNN': KNeighborsClassifier(),
                'XGBoost': XGBClassifier(),
                'GradientBoost': GradientBoostingClassifier(),
                'AdaBoost': AdaBoostClassifier()
            }

            # Evaluate models and get report
            model_report, best_model_name = evaluate_model(X_train, X_test, y_train, y_test, models)

            # Log model metrics
            logging.info("All Models' Metrics:")
            for model_name, metrics in model_report.items():
                logging.info(f"Model: {model_name}")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value}")
                logging.info("-" * 60)

            # Log and save best model based on Test F1-Score
            logging.info(f"Best Model: {best_model_name} with Test F1-Score: {model_report[best_model_name]['test_f1']}")
            best_model = models[best_model_name]
            save_obj('best_model.pkl', best_model)

            logging.info("Model Training Completed")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)