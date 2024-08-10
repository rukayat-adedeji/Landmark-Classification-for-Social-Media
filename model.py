import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:

        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.features = nn.Sequential(
            # First Convolutional layer
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # -> 16x224x224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 16x112x112
            
            # Second Convolutional layer
            nn.Conv2d(16, 32, 3, padding=1),  # -> 32x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            
            # Third Convolutional layer
            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x28x28
            
            # Fourth Convolutional layer
            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x14x14
        )
            
        self.classifier = nn.Sequential(
            nn.Flatten(),  # -> 1x128x14x14
            
            # Fully Connected Layers
            # FC1
            nn.Linear(128 * 14 * 14, 2048),  # -> 13044
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # FC2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # FC3
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.features(x)
        x = self.classifier(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
