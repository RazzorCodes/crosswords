try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

if nn is not None:
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=26):
            super(SimpleCNN, self).__init__()
            # Smaller filter counts for efficiency
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # Input 64x64 -> conv1 -> pool -> 32x32
            # 32x32 -> conv2 -> pool -> 16x16
            # 16x16 -> conv3 -> pool -> 8x8
            self.fc1 = nn.Linear(32 * 8 * 8, 64)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 32 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
else:
    class SimpleCNN:
        def __init__(self, *args, **kwargs):
            pass

def get_cnn_model(num_classes=26):
    return SimpleCNN(num_classes)
