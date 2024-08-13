from common import *

from sys import argv

# Extract features from images
image_model = FrozenDinoModel()

# Load the weights
image_model.load_state_dict(torch.load(f'image_model_frozendino.pth', device))

# Extract the features
features = extract_features(image_model, f'image_model_frozendino_features.csv')
val_features = extract_test_features(image_model, f'image_model_frozendino_testfeatures.csv')


