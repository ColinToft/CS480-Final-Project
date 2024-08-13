import torch

from sklearn.metrics import mean_squared_error, r2_score

from common import *
from scipy.stats import skew

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Params for all 6 lgbms, found through optuna using optuna_lgbm_combined.py
params = [
    {'n_estimators': 1333, 'learning_rate': 0.07347245808014251, 'reg_alpha': 0.48876733817608126, 'reg_lambda': 0.4689370288213431, 'colsample_bytree': 0.6621280128098435, 'num_leaves': 63, 'max_depth': 27, 'numerical_model_version': 'best'},
    {'n_estimators': 1470, 'learning_rate': 0.04116386077399362, 'reg_alpha': 0.45283038363893713, 'reg_lambda': 0.6735876674091854, 'colsample_bytree': 0.4367954345790499, 'num_leaves': 62, 'max_depth': 52, 'numerical_model_version': 'best'},
    {'n_estimators': 1420, 'learning_rate': 0.03918863937972138, 'reg_alpha': 0.4788168758299612, 'reg_lambda': 0.1483718807915807, 'colsample_bytree': 0.5154256470022031, 'num_leaves': 61, 'max_depth': 28, 'numerical_model_version': 'best'},
    {'n_estimators': 1399, 'learning_rate': 0.0451635532813527, 'reg_alpha': 0.6065470344870729, 'reg_lambda': 0.6547041680073944, 'colsample_bytree': 0.15047125473448275, 'num_leaves': 60, 'max_depth': 32, 'numerical_model_version': None},
    {'n_estimators': 1467, 'learning_rate': 0.04832141625096779, 'reg_alpha': 0.3478672354329516, 'reg_lambda': 0.934255101460429, 'colsample_bytree': 0.4262090996171284, 'num_leaves': 55, 'max_depth': 22, 'numerical_model_version': 'best'},
    {'n_estimators': 1457, 'learning_rate': 0.0349741865251378, 'reg_alpha': 0.46266415428259305, 'reg_lambda': 0.6551236138367577, 'colsample_bytree': 0.5162765634984653, 'num_leaves': 62, 'max_depth': 48, 'numerical_model_version': 'best'}
]
models = [None] * 6
r2s = [None] * 6

for selected_label in range(6):
    extra_csvs = []

    if params[selected_label]['numerical_model_version'] is not None:
        extra_csvs.append(f'numerical_model_features_{selected_label}.csv')
    image_model_version = 'frozendino'
    if image_model_version is not None:
        extra_csvs.append(f'image_model_{image_model_version}_features.csv')

    normalizers, train_dataset, test_dataset = load_combined_datasets('numerical', f'{FILEPATH}/train.csv', f'{FILEPATH}/train_images', selected_label=selected_label, normalized=False, extra_csvs=extra_csvs)

    print(train_dataset.data.shape)

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.labels[:, 0].numpy() # Converts length 1 tensor to scalar

    X_test = test_dataset.data.numpy()
    y_test = test_dataset.labels[:, 0].numpy() # Converts length 1 tensor to scalar

    models[selected_label] = train_lgbm_regressor(X_train, y_train, n_estimators=params[selected_label]['n_estimators'], learning_rate=params[selected_label]['learning_rate'], reg_alpha=params[selected_label]['reg_alpha'], reg_lambda=params[selected_label]['reg_lambda'], colsample_bytree=params[selected_label]['colsample_bytree'], num_leaves=params[selected_label]['num_leaves'], max_depth=params[selected_label]['max_depth'])

    # Evaluate the model
    y_pred = models[selected_label].predict(X_test)
    r2s[selected_label] = r2_score(y_test, y_pred)
    print(f'Model {selected_label} R^2: {r2s[selected_label]}')

print(f"Overall R^2: {np.mean(r2s)}")

# Submission
predictions = None

for selected_label in range(6):
    extra_csvs = []

    if params[selected_label]['numerical_model_version'] is not None:
        extra_csvs.append(f'numerical_model_testfeatures_{selected_label}.csv')
    image_model_version = 'frozendino'
    if image_model_version is not None:
        extra_csvs.append(f'image_model_{image_model_version}_testfeatures.csv')

    normalizers, validation_dataset, _ = load_combined_datasets('numerical', f'{FILEPATH}/test.csv', f'{FILEPATH}/test_images', normalized=False, extra_csvs=extra_csvs, is_test=True)

    if predictions is None:
        predictions = np.zeros((len(validation_dataset), 6))

    X_validation = validation_dataset.data.numpy()

    predictions[:, selected_label] = models[selected_label].predict(X_validation)

image_names = validation_dataset.image_names

# Write the csv with the correct header, then write the predictions
with open(f'combined_submission.csv', 'w') as f:
    HEADER = 'id,X4,X11,X18,X26,X50,X3112'

    f.write(HEADER + '\n')

    for i in range(len(predictions)):
        f.write(f'{image_names[i]},{",".join(map(str, predictions[i]))}\n')
