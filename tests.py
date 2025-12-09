from config import Config
import numpy as np
import nabla.nn as nn

config = Config()
seed = config.seed
np.random.seed(seed=seed)

def test_regression():
    print('Regression test')
    x = np.random.randn(config.batch_size, config.num_samples, config.in_features) # (batch_size, num_samples, num_features)
    x = x.reshape(-1, config.in_features) # Flatten to (total_samples, num_features)

    y = np.random.randn(config.batch_size, config.num_samples, 1) # (batch_size, num_samples)
    y = y.reshape(-1, 1) # Flatten to (total_samples, 1)

    print(f'Input shape: {x.shape}')
    print(f'Label shape: {y.shape}')

    # Linear -> ReLU -> Linear
    mlp = [nn.Linear(config.in_features, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, 1)]

    criterion = nn.MSELoss()

    for layer in mlp:
        x = layer.forward(x)

    loss = criterion.forward(x, y)

    print(f'Output shape {x.shape}')
    print(f'MSE Loss: {loss}')

def test_classification():
    print('Classification test')
    x = np.random.randn(config.batch_size, config.num_samples, config.in_features)
    x = x.reshape(-1, config.in_features)

    y = np.random.randint(0, config.num_classes, size=(config.batch_size * config.num_samples,))

    print(f'Input shape: {x.shape}')
    print(f'Label shape: {y.shape}')

    mlp = [nn.Linear(config.in_features, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, config.num_classes)]
    criterion = nn.CrossEntropyLoss()

    for layer in mlp:
        x = layer.forward(x)

    print(f'Logit shape: {x.shape}')

    loss = criterion.forward(x, y)
    probs = criterion.probabilities

    print(f'Probability sum: {np.sum(probs,axis=1)}')

    print(f'Cross entropy loss: {loss}')

    expected_loss = np.log(config.num_classes)
    print(f'Expected loss: {expected_loss}')