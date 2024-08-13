import torch

from sklearn.metrics import mean_squared_error, r2_score

from common import *
from scipy.stats import skew
import optuna

from sys import argv

selected_label = int(argv[1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lgbm_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.15, 1)
    num_leaves = trial.suggest_int('num_leaves', 2, 64)
    max_depth = trial.suggest_int('max_depth', 2, 64)

    numerical_model_version = trial.suggest_categorical('numerical_model_version', [None, 'best'])
    # image_model_version = trial.suggest_categorical('image_model_version', ['resnext', 'swin'])
    image_model_version = 'frozendino'

    extra_csvs = []

    if numerical_model_version is not None:
        extra_csvs.append(f'numerical_model_features_{selected_label}.csv')
    if image_model_version is not None:
        extra_csvs.append(f'image_model_{image_model_version}_features.csv')

    normalizers, train_dataset, test_dataset = load_combined_datasets('numerical', f'{FILEPATH}/train.csv', f'{FILEPATH}/train_images', selected_label=selected_label, normalized=False, extra_csvs=extra_csvs)

    print(train_dataset.data.shape)
    print(train_dataset.labels.shape)

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.labels[:, 0].numpy() # Converts length 1 tensor to scalar

    X_test = test_dataset.data.numpy()
    y_test = test_dataset.labels[:, 0].numpy() # Converts length 1 tensor to scalar
    
    model = train_lgbm_regressor(X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda, colsample_bytree=colsample_bytree, num_leaves=num_leaves, max_depth=max_depth)
    
    return r2_score(y_test, model.predict(X_test))

study = optuna.create_study(direction='maximize')
study.optimize(lgbm_objective, n_trials=500)

print(study.best_params)
print(study.best_value)
