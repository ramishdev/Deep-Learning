import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.batch_dim = None

    def forward(self, input_tensor):
        self.batch_dim = input_tensor.shape[0]
        return input_tensor.reshape(self.batch_dim, -1)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_shape, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.residual_conv = (in_channels != out_channels or stride_shape != 1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential(
            self.conv1,
            self.batch_norm1,
            self.relu,
            self.conv2,
            self.batch_norm2
        )

    def forward(self, input_tensor):
        residual = input_tensor
        output_tensor = self.skip(input_tensor)
        if self.residual_conv:
            residual = self.conv1x1(residual)
        residual = self.batch_norm3(residual)
        
        output_tensor += residual
        output_tensor = self.relu(output_tensor)
        
        return output_tensor

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(in_channels=64, out_channels=64),
            ResBlock(in_channels=64, out_channels=128, stride_shape=2),
            ResBlock(in_channels=128, out_channels=256, stride_shape=2),
            nn.Dropout(p=0.5),
            ResBlock(in_channels=256, out_channels=512, stride_shape=2),
            nn.AvgPool2d(kernel_size=10),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        output_tensor = self.seq1(input_tensor)
        return output_tensor