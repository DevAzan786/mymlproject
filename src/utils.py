import sys
import os
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
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
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        report = {}
        best_model = None
        best_f1_score = -1

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics for training and test sets
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Store results
            report[model_name] = {
                'train_f1': train_f1,
                'test_f1': test_f1,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'accuracy_difference': abs(train_accuracy - test_accuracy)
            }

            if test_f1 > best_f1_score:
                best_f1_score = test_f1
                best_model = model_name

        return report, best_model

    except Exception as e:
        raise CustomException(e, sys)
