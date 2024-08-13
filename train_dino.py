from common import *

from sys import argv

from torch import nn
from torch.optim import AdamW, lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_model = FrozenDinoModel().to(device)

criterion = nn.MSELoss()
optimizer = AdamW(image_model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

train_model('image', image_model, criterion, optimizer, scheduler, batch_size=32, epochs=50)
