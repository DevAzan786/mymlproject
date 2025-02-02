import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exception import CustomException


def save_obj(file_path, obj):
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Save the object
        with open(file_path, 'wb') as f:
            joblib.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, X_test, y_train, y_test, models, param, accuracy_threshold=0.8):
    try:
        report = {}
        best_model = None
        best_f1_score = -1

        for model_name, model in models.items():
            c_v = KFold(5)
            if model_name in param:
                # Perform RandomizedSearchCV for hyperparameter tuning
                grid_search = RandomizedSearchCV(estimator=model, param_distributions=param[model_name],
                                                 n_iter=30, cv=c_v, n_jobs=-1, verbose=2, random_state=42)

                grid_search.fit(X_train, y_train)
                tuned_model = grid_search.best_estimator_

                y_train_pred = tuned_model.predict(X_train)
                y_test_pred = tuned_model.predict(X_test)

                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')

                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                report[model_name] = {
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'accuracy_difference': abs(train_accuracy - test_accuracy)
                }

                # Select the best model based on F1 score and accuracy threshold
                if test_f1 > best_f1_score and test_accuracy >= accuracy_threshold:
                    best_f1_score = test_f1
                    best_model = model_name


            if best_model is None:
                best_model = max(report, key=lambda k: report[k]['test_f1'], default=None)

        return report, best_model

    except Exception as e:
        raise CustomException(e, sys)
