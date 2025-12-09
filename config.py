from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42

    batch_size: int = 16
    num_samples: int = 10
    in_features: int = 5
    num_classes: int = 3

    hidden_size: int = 5
