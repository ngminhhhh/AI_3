import torch, torch.nn as nn

class Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return self.relu(out)
    
class MiniChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[Block(64) for _ in range(3)]
        )
        self.policy = nn.Linear(64*8*8, 4672)
        self.value  = nn.Linear(64*8*8, 1)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return self.policy(x), torch.tanh(self.value(x))
    
