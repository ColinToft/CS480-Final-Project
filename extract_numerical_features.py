from common import *

import optuna

# Read command line args
from sys import argv

selected_label = int(argv[1])
EPOCHS = 30

numerical_model = NumericalModel(output_size=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(numerical_model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Use optuna to search for the best lower and upper quantiles
def objective(trial):
    # Define the search space
    lower_quantile_selected = trial.suggest_float('lower_quantile', 0.00001, 0.02, log=True)
    upper_quantile_selected = trial.suggest_float('upper_quantile', 0.98, 0.99999, log=True)

    lower_quantile_overall = trial.suggest_float('lower_quantile_overall', 0.00001, 0.02, log=True)
    upper_quantile_overall = trial.suggest_float('upper_quantile_overall', 0.98, 0.99999, log=True)
    
    # Train the model
    _, weights, loss = train_model('numerical', numerical_model, criterion, optimizer, scheduler, batch_size=256, epochs=EPOCHS, selected_label=selected_label, lower_quantiles=[lower_quantile_selected, lower_quantile_overall], upper_quantiles=[upper_quantile_selected, upper_quantile_overall])

    # Save the weights if the loss is lower
    try: 
        if loss < study.best_trial.value:
            torch.save(weights, f'numerical_model_best_{selected_label}.pth')
    except ValueError:
        # This is the first trial
        torch.save(weights, f'numerical_model_best_{selected_label}.pth')

    return loss

TRAIN = True

if TRAIN:
    # Search for a model that minimizes the loss
    _, best_weights, baseline_loss = train_model('numerical', numerical_model, criterion, optimizer, scheduler, batch_size=256, epochs=EPOCHS, selected_label=selected_label, lower_quantiles=[0] * 2, upper_quantiles=[1] * 2)

    # Save baseline weights
    torch.save(best_weights, f'numerical_model_baseline_{selected_label}.pth')
    print("Baseline loss: ", baseline_loss)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    if study.best_trial.value < baseline_loss:
        # Save the best weights
        torch.save(best_weights, f'numerical_model_best_{selected_label}.pth')

    print("Best trial:")
    print(study.best_trial.params)
    print(study.best_trial.value)

    print("Baseline loss: ", baseline_loss)

# Load the best weights and extract the features
numerical_model.load_state_dict(torch.load(f'numerical_model_best_{selected_label}.pth', weights_only=True))
numerical_model.eval()

# Extract the features
features = extract_features(numerical_model, f'numerical_model_features_{selected_label}.csv')
features_test = extract_test_features(numerical_model, f'numerical_model_testfeatures_{selected_label}.csv')


