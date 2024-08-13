import torch
import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, RandomRotation, RandomResizedCrop, ToTensor, Resize
from scipy.stats import skew

from sklearn.metrics import r2_score
from torch import nn

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_FEATURES = 6
INPUT_FEATURES = 163
IMAGE_SIZE = 128

IMAGE_OUTPUT = 64

NUMERICAL_DROPOUT = 0.5

FILEPATH = 'data'
OUTPUT_PATH = FILEPATH

runtime = None
try:
    import google.colab
    runtime = "COLAB"
except:
    pass

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE',''):
    runtime = "KAGGLE"

if runtime == "COLAB":
    FILEPATH = '/tmp/data'
    OUTPUT_PATH = '/content/drive/MyDrive/PlantPredictor'
elif runtime == "KAGGLE":
    FILEPATH = '/kaggle/input/cs-480-2024-spring/data'
    OUTPUT_PATH = '/kaggle/working'


class PlantDataset(Dataset):
    def __init__(self, mode, image_names, data, labels, image_folder, image_transform=None, numerical_transform=None):
        self.mode = mode
        self.image_names = image_names
        self.data = data
        self.labels = labels
        self.image_folder = image_folder
        self.image_transform = image_transform
        self.numerical_transform = numerical_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'image' or self.mode == 'both':
            img_name = f'{self.image_folder}/{self.image_names[idx]}.jpeg'
            image = Image.open(img_name)

            if self.image_transform:
                image = self.image_transform(image)
            
                # Make sure the data is float
                image = image.float()
        else:
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        # Get the numerical data
        numerical_data = self.data[idx]

        if self.numerical_transform:
            numerical_data = self.numerical_transform(numerical_data)

        assert numerical_data.shape[0] == INPUT_FEATURES

        return image, numerical_data, self.labels[idx]


def load_datasets(mode, filename, image_folder, lower_quantiles=[0, 0], upper_quantiles=[1, 1], selected_label=None, normalized=True):
    # Load the CSV file
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    np.random.seed(0)
    np.random.shuffle(data)

    image_names = data[:, 0].astype(int).astype(str)
    data = data[:, 1:]

    print("Data shape:", data.shape)


    # We need to split the training dataset into a training and testing dataset
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    # Normalize the data using the Normalizer class
    normalizers = []
    for i in range(data.shape[1]):
        normalizer = Normalizer(data[:train_size, i], enabled=normalized)
        data[:, i] = normalizer(data[:, i])
        normalizers.append(normalizer)

    # Extract the labels
    labels = data[:, -OUTPUT_FEATURES:]
    data = data[:, :-OUTPUT_FEATURES]

    # Confirm shapes
    print(data.shape)
    print(data.shape[1])
    assert data.shape[1] == INPUT_FEATURES

    if selected_label is not None:
        selected_training_data = filter_data(labels[:train_size], selected_label, lower_quantiles, upper_quantiles)
        labels = labels[:, selected_label:selected_label+1]
        assert labels.shape == (len(data), 1)
    else:
        assert labels.shape[1] == OUTPUT_FEATURES
        selected_training_data = slice(None) # All data

    data = torch.tensor(data).float()
    labels = torch.tensor(labels).float()

    # Make a random split of the data, and remember to use data augmentation on the training dataset
    train_dataset = PlantDataset(mode, image_names[:train_size][selected_training_data], data[:train_size][selected_training_data], labels[:train_size][selected_training_data], image_folder, image_transform=train_image_transform, numerical_transform=train_numerical_transform)
    test_dataset = PlantDataset(mode, image_names[train_size:], data[train_size:], labels[train_size:], image_folder, image_transform=test_image_transform, numerical_transform=test_numerical_transform)

    return normalizers, train_dataset, test_dataset

def load_combined_datasets(mode, filename, image_folder, lower_quantiles=[0, 0], upper_quantiles=[1, 1], selected_label=None, normalized=True, extra_csvs=[], is_test=False):
    """
    Same as load_datasets, but optionally can add additional CSV files of features.
    The first column of the CSV file should be the image name, and the rest should be the features.
    """
    extra_cols = 0 if is_test else OUTPUT_FEATURES

    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    image_names = data[:, 0].astype(int).astype(str)
    for csv_file in extra_csvs:
        extra_data = np.loadtxt(csv_file, delimiter=',', skiprows=1)

        image_names_new = extra_data[:, 0].astype(int).astype(str)
        assert np.array_equal(image_names, image_names_new)

        extra_data = extra_data[:, 1:]

        if is_test:
            data = np.concatenate([data, extra_data], axis=1)
        else:
            # Insert in the extra data, but before the labels
            data = np.concatenate([data[:, :-extra_cols], extra_data, data[:, -extra_cols:]], axis=1)

    if not is_test:
        np.random.seed(0)
        np.random.shuffle(data)

    data = data[:, 1:]

    print("Data shape:", data.shape)

    # We need to split the training dataset into a training and testing dataset
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    # Normalize the data using the Normalizer class
    normalizers = []
    for i in range(data.shape[1]):
        normalizer = Normalizer(data[:train_size, i], enabled=normalized)
        data[:, i] = normalizer(data[:, i])
        normalizers.append(normalizer)

    # Extract the labels
    if not is_test:
        labels = data[:, -OUTPUT_FEATURES:]
        data = data[:, :-OUTPUT_FEATURES]

        if selected_label is not None:
            selected_training_data = filter_data(labels[:train_size], selected_label, lower_quantiles, upper_quantiles)
            labels = labels[:, selected_label:selected_label+1]
            assert labels.shape == (len(data), 1)
        else:
            assert labels.shape[1] == OUTPUT_FEATURES
            selected_training_data = slice(None) # All data

        data = torch.tensor(data).float()
        labels = torch.tensor(labels).float()

        # Make a random split of the data, and remember to use data augmentation on the training dataset
        train_dataset = PlantDataset(mode, image_names[:train_size][selected_training_data], data[:train_size][selected_training_data], labels[:train_size][selected_training_data], image_folder, image_transform=train_image_transform, numerical_transform=train_numerical_transform)
        test_dataset = PlantDataset(mode, image_names[train_size:], data[train_size:], labels[train_size:], image_folder, image_transform=test_image_transform, numerical_transform=test_numerical_transform)

        return normalizers, train_dataset, test_dataset

    # Otherwise, this dataset is a validation dataset
    else:
        data = torch.tensor(data).float()
        fake_labels = np.zeros((len(data), OUTPUT_FEATURES))
        return normalizers, PlantDataset(mode, image_names, data, fake_labels, image_folder, image_transform=test_image_transform, numerical_transform=test_numerical_transform), None


def filter_data(labels, selected_label, lower_quantiles, upper_quantiles):
    # Filters the data based on the lower and upper quantiles of the labels
    # Return the indices of the data that is within the quantiles
    # Quantiles are an array of 2 values, one for the selected_label and one for the overall, which uses all labels and Mahalanobis distance
    # The quantiles are in the range [0, 1]
    # Quantiles 0 are the quantile bounds for each label
    # Quantiles 1 is the upper and lower quantile for Mahalanobis distance

    print(f"Filtering data with lower quantiles {lower_quantiles} and upper quantiles {upper_quantiles}")

    # Calculate the Mahalanobis distance
    mean = np.mean(labels, axis=0)
    cov = np.cov(labels, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    if lower_quantiles[-1] != 0 or upper_quantiles[-1] != 1:
        mahalanobis = np.sum((labels - mean) @ inv_cov * (labels - mean), axis=1)
        mahalanobis = np.sqrt(mahalanobis)

        assert mahalanobis.shape[0] == len(labels)

        # Calculate the quantiles
        lower_quantile = np.quantile(mahalanobis, lower_quantiles[-1])
        upper_quantile = np.quantile(mahalanobis, upper_quantiles[-1])

        # Filter the data
        indices = (mahalanobis > lower_quantile) & (mahalanobis < upper_quantile)
    else:
        indices = np.ones(len(labels), dtype=bool)


    if lower_quantiles[0] != 0 or upper_quantiles[0] != 1:
        lower_quantile = np.quantile(labels[:, selected_label], lower_quantiles[0])
        upper_quantile = np.quantile(labels[:, selected_label], upper_quantiles[0])

        indices &= (labels[:, selected_label] > lower_quantile) & (labels[:, selected_label] < upper_quantile)

    amount_filtered = len(labels) - np.sum(indices).item()
    print(f'Filtered {amount_filtered} out of {len(labels)} samples')

    return indices

class Normalizer():
    """
    Takes in a column of data and normalizes it to have a mean of 0 and a standard deviation of 1.
    Can optionally log transform the data to make it more normal.
    """

    def __init__(self, data, minmax=False, enabled=True):
        self.enabled = enabled
        self.log_transforms = []

        MAX_LOGS = 0

        for i in range(MAX_LOGS):
            if skew(data) > 1.3:
                self.log_transforms.append(np.min(data))
                data = np.log1p(data - np.min(data) + 1)
            else:
                break

        self.mean = np.mean(data)
        self.std = np.std(data)

        self.min = np.min(data)
        self.max = np.max(data)

        self.minmax = minmax

    def __call__(self, x):
        if not self.enabled:
            return x

        if self.minmax:
            x = (x - self.min) / (self.max - self.min)
        else:
            for i in range(len(self.log_transforms)):
                # Make sure no values are smaller than self.log_transforms[i] to avoid negative values
                x = np.maximum(x, self.log_transforms[i])

                x = np.log1p(x - self.log_transforms[i] + 1)

            x = (x - self.mean) / self.std

        return x

    def inverse(self, x):
        if not self.enabled:
            return x

        if self.minmax:
            x = x * (self.max - self.min) + self.min
        
        else:
            x = x * self.std + self.mean

            for i in range(len(self.log_transforms) - 1, -1, -1):
                x = np.expm1(x) + self.log_transforms[i] - 1

        return x


# Note, the below augmentations will also scale the images from 128 to 224
data_augmentation = transforms.Compose([
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    RandomRotation(15),
    RandomResizedCrop(224, scale=(0.8, 1.0)),
])

train_image_transform = transforms.Compose([
    data_augmentation,
    ToTensor()
])

test_image_transform = transforms.Compose([
    Resize(224),
    ToTensor()
])

train_numerical_transform = transforms.Compose([
    # Uncomment and edit to add a bit of noise to the data
    # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.0),
])

test_numerical_transform = transforms.Compose([
])

def train_lgbm_regressor(X, y, n_estimators=1000, learning_rate=0.01, reg_alpha=0, reg_lambda=0, colsample_bytree=1, num_leaves=31, max_depth=-1):
    model = LGBMRegressor(verbosity=-1, n_estimators=n_estimators, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda, colsample_bytree=colsample_bytree, num_leaves=num_leaves, max_depth=max_depth)
    model.fit(X, y)
    return model    

class ResnextModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Time for the big guns (resnext50_32x4d)
        self.cnn = models.resnext50_32x4d(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, IMAGE_OUTPUT)
        self.final = nn.Linear(IMAGE_OUTPUT, OUTPUT_FEATURES)
        
    def forward(self, x):
        features = self.cnn(x)
        return self.final(features)
    
    def extract_features(self, images, data):
        return self.cnn(images)
    
    def get_output_size(self):
        return IMAGE_OUTPUT

    def get_input_mode(self):
        return 'image'

class ResnextTransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=6):
        super(ResnextTransformerModel, self).__init__()
        self.d_model = d_model

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Load resnext50_32x4d pretrained model
        self.cnn = models.resnext50_32x4d(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, d_model)

        self.fc = nn.Linear(d_model, OUTPUT_FEATURES)

    def forward(self, image, numerical):
        # Put image through resnext50_32x4d and then transformer on the output
        image = self.cnn(image)
    
        output = self.transformer(image)

        return self.fc(output)
    
    def extract_features(self, image, numerical):
        # Put image through resnext50_32x4d and then transformer on the output
        image = self.cnn(image)
    
        return self.transformer(image)

class SwinTransformerModel(nn.Module):
    def __init__(self):
        super(SwinTransformerModel, self).__init__()

        self.transformer = models.swin_transformer.swin_v2_b(weights=models.swin_transformer.Swin_V2_B_Weights.IMAGENET1K_V1)
        second_last_layer_size = self.transformer.head.in_features

        self.transformer.head = nn.Linear(second_last_layer_size, IMAGE_OUTPUT)

        self.final = nn.Linear(IMAGE_OUTPUT, OUTPUT_FEATURES)
        
    def forward(self, image, numerical):
        features = self.transformer(image)
        return self.final(features)
    
    def extract_features(self, image, numerical):
        return self.transformer(image)

class VitModel(nn.Module):
    def __init__(self):
        super(VitModel, self).__init__()

        self.transformer = models.vision_transformer.vit_l_16(image_size=128)
        second_last_layer_size = self.transformer.heads.head.in_features

        self.transformer.heads.head = nn.Linear(second_last_layer_size, IMAGE_OUTPUT)

        self.final = nn.Linear(IMAGE_OUTPUT, OUTPUT_FEATURES)
        
    def forward(self, image, numerical):
        features = self.transformer(image)
        return self.final(features)
    
    def extract_features(self, image, numerical):
        return self.transformer(image)

class DinoModel(nn.Module):
    def __init__(self):
        super(DinoModel, self).__init__()

        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        second_last_layer_size = 384

        self.final = nn.Linear(second_last_layer_size, OUTPUT_FEATURES)
        
    def forward(self, image, numerical):
        features = self.transformer(image)
        features = self.transformer.norm(features)
        return self.final(features)
    
    def extract_features(self, image, numerical):
        return self.transformer(image)

class FrozenDinoModel(nn.Module):
    def __init__(self):
        super(FrozenDinoModel, self).__init__()

        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
        second_last_layer_size = 1024

        self.head = nn.Sequential(
            nn.Linear(second_last_layer_size, 64),
            nn.SELU(),
            nn.BatchNorm1d(64)
        )

        self.final = nn.Linear(64, OUTPUT_FEATURES)

        for param in self.transformer.parameters():
            param.requires_grad = False
        
    def forward(self, image, numerical):
        features = self.transformer(image)
        features = self.transformer.norm(features)
        features = self.head(features)
        return self.final(features)
    
    def extract_features(self, image, numerical):
        features = self.transformer(image)
        features = self.transformer.norm(features)
        return self.head(features)

class NumericalModel(nn.Module):
    def __init__(self, output_size=OUTPUT_FEATURES):
        super(NumericalModel, self).__init__()

        numerical_layer_sizes = [INPUT_FEATURES, 256, 64, output_size]
        
        layers = []
        for i in range(len(numerical_layer_sizes) - 2):
            layers.append(nn.Linear(numerical_layer_sizes[i], numerical_layer_sizes[i+1]))
            layers.append(nn.SELU())
            layers.append(nn.BatchNorm1d(numerical_layer_sizes[i+1]))

        self.fc_numerical = nn.Sequential(*layers)
        self.final = nn.Linear(numerical_layer_sizes[-2], output_size)

    def extract_features(self, images, data):
        return self.fc_numerical(data)

    def forward(self, image, numerical):
        x = self.fc_numerical(numerical)
        x = self.final(x)

        return x

def generate_submission(mode, model, normalizers, filename, image_folder, output_filename):
    model.eval()
    HEADER = 'id,X4,X11,X18,X26,X50,X3112'

    # Load test data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    image_names = data[:, 0].astype(int).astype(str)
    data = data[:, 1:]

    assert len(normalizers) == data.shape[1] + OUTPUT_FEATURES
    assert data.shape[1] == INPUT_FEATURES

    # Normalize the data
    for i in range(data.shape[1]):
        data[:, i] = normalizers[i](data[:, i])

    data = torch.tensor(data).float()

    labels = np.zeros((len(data), OUTPUT_FEATURES))

    dataset = PlantDataset(mode, image_names, data, labels, image_folder, image_transform=test_image_transform, numerical_transform=test_numerical_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Write the predictions to a CSV file

    rows_written = 0 
    with open(output_filename, 'w') as f:
        f.write(HEADER + '\n')
        for images, batch_data, _ in loader:
            images = images.to(device)
            batch_data = batch_data.to(device)

            output = model(images, batch_data)

            output = output.cpu().detach().numpy()
            for i in range(output.shape[1]):
                output[:, i] = normalizers[INPUT_FEATURES + i].inverse(output[:, i])

            for row in output:
                f.write(f'{image_names[rows_written]},' + ','.join(map(str, row)) + '\n')
                rows_written += 1

    assert rows_written == len(data)

    return output_filename


def write_train_predictions(model, normalizers, output_path):
    return generate_submission(model, normalizers, output_path, f'{FILEPATH}/train.csv', f'{FILEPATH}/train_images')

def extract_features(model, output_filename, feature_count=IMAGE_OUTPUT):
    # Calls model to extract features and write them all to a file
    model.eval()
    
    filename = f'{FILEPATH}/train.csv'
    image_folder = f'{FILEPATH}/train_images'

    # Load the data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    image_names = data[:, 0].astype(int).astype(str)
    data = data[:, 1:]

    data = torch.tensor(data).float()

    # Extract the labels
    labels = data[:, -OUTPUT_FEATURES:]
    data = data[:, :-OUTPUT_FEATURES]

    dataset = PlantDataset('both', image_names, data, labels, image_folder, image_transform=test_image_transform, numerical_transform=test_numerical_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    feature_file = open(output_filename, 'w')
    feature_file.write('id,' + ','.join([f'feature{i}' for i in range(feature_count)]) + '\n')

    rows_written = 0
    for images, batch_data, _ in tqdm(loader):
        images = images.to(device)
        batch_data = batch_data.to(device)

        features = model.extract_features(images, batch_data)

        features = features.cpu().detach().numpy()

        # features has shape (batch_size, feature_count)
        # Write each row to the file, each row will have length feature_count
        assert features.shape == (len(images), feature_count)

        for row in features:
            feature_file.write(f'{image_names[rows_written]},' + ','.join(map(str, row)) + '\n')
            rows_written += 1

    assert rows_written == len(data)

    feature_file.close()

def extract_test_features(model, output_filename, feature_count=IMAGE_OUTPUT):
    # Calls model to extract features and write them all to a file
    model.eval()
    
    filename = f'{FILEPATH}/test.csv'
    image_folder = f'{FILEPATH}/test_images'

    # Load the data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    image_names = data[:, 0].astype(int).astype(str)
    data = data[:, 1:]

    data = torch.tensor(data).float()

    fake_labels = np.zeros((len(data), OUTPUT_FEATURES))

    dataset = PlantDataset('both', image_names, data, fake_labels, image_folder, image_transform=test_image_transform, numerical_transform=test_numerical_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    feature_file = open(output_filename, 'w')
    feature_file.write('id,' + ','.join([f'feature{i}' for i in range(feature_count)]) + '\n')

    rows_written = 0
    for images, batch_data, _ in tqdm(loader):
        images = images.to(device)
        batch_data = batch_data.to(device)

        with torch.no_grad():
            features = model.extract_features(images, batch_data)
            features = features.cpu().detach().numpy()

            # features has shape (batch_size, feature_count)
            # Write each row to the file, each row will have length feature_count
            assert features.shape == (len(images), feature_count)

            for row in features:
                feature_file.write(f'{image_names[rows_written]},' + ','.join(map(str, row)) + '\n')
                rows_written += 1

    assert rows_written == len(data)

    feature_file.close()


def train_model(mode, model, criterion, optimizer, scheduler, batch_size=256, epochs=50, lower_quantiles=[0, 0], upper_quantiles=[1, 1], selected_label=None, normalized=True):
    normalizers, train_dataset, test_dataset = load_datasets(mode, f'{FILEPATH}/train.csv', f'{FILEPATH}/train_images', lower_quantiles, upper_quantiles, selected_label=selected_label, normalized=normalized)
    # verification_dataset = load_dataset(f'{FILEPATH}/test.csv', f'{FILEPATH}/test_images')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model and graph the loss
    train_losses = []
    test_losses = []


    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, data, labels in tqdm(train_loader):
            images = images.to(device)
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images, data)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(f'Epoch {epoch} - Train Loss: {train_loss}', end=', ')

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, data, labels in test_loader:
                images = images.to(device)
                data = data.to(device)
                labels = labels.to(device)

                output = model(images, data)
                loss = criterion(output, labels)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f'Test Loss: {test_loss}')

        scheduler.step(test_loss)

        output_path = f'{OUTPUT_PATH}/submission-{epoch}.csv'

        generate_submission(mode, model, normalizers, f'{FILEPATH}/test.csv', f'{FILEPATH}/test_images', output_path)

        # Save graph so far to a file
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.legend()
        plt.savefig(f'{OUTPUT_PATH}/loss.png')
        plt.close()

        # If this is the best model so far, save the weights
        if test_loss == min(test_losses):
            torch.save(model.state_dict(), f'{OUTPUT_PATH}/model-{epoch}.pth')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    # plt.show()

    return model, weights, min(test_losses)
