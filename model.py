from torch import nn


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Первый свёрточный блок работает с исходной картинкой 1x28x28.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Второй блок увеличивает количество каналов признаков.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Третий блок выделяет более сложные признаки цифры.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Общие слои используются после каждого свёрточного блока.
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_conv = nn.Dropout(p=0.3)
        self.dropout_fc = nn.Dropout(p=0.5)

        # После трёх MaxPool размер 28x28 превращается в 3x3.
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # forward описывает полный проход изображения через сеть.
        # Первый блок: 28x28 -> 14x14.
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Второй блок: 14x14 -> 7x7.
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Третий блок: 7x7 -> 3x3.
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Полносвязная часть превращает признаки в 10 классов.
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
