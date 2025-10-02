import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        xin = x
        out = self.conv2(self.relu(self.bn1(self.conv1(x))))
        conx2 = torch.cat((xin, out), dim=1)
        return out, conx2


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, conx):
        xin = x
        out = self.conv(self.relu(self.bn1(x)))
        conx2 = torch.cat((conx, out), dim=1)
        return xin + out, conx2

class TransLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TransLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, conx):
        out = self.relu(self.bn(self.conv(conx)))
        return out


class LDB(nn.Module):
    def __init__(self, in_panel: int, t: float = 0.5):
        super(LDB, self).__init__()
        self.layer1 = Bottleneck(in_panel, int(in_panel*t))
        self.basicblock1 = BasicBlock(int(in_panel*t), int(in_panel*t))


    def forward(self, x):
        x_sum, conx = self.layer1(x)
        x_sum, conx = self.basicblock1(x_sum, conx)
        x_sum, conx = self.basicblock1(x_sum, conx)
        x_sum, conx = self.basicblock1(x_sum, conx)
        return conx


class CDenseNet(nn.Module):
  def __init__(self, n: int = 16, t: float = 0.5, num_class: int = 3, C0 = 32):
    super(CDenseNet, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(1, C0, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(C0),
        nn.ReLU()
    )
    self.dense_blocks = nn.ModuleList()
    current_channels = C0

    for i in range(16):  # 重複 16 次

        # LDB 輸入通道數
        C_LDB_in = current_channels

        # LDB 輸出通道數 (Conx): C_in + B * k。由於 LDB 內部 BasicBlock 使用的通道是 k=16，
        # LDB 內的 BasicBlock 輸出 channel= in*t = 16
        k = int(C0 * t)
        C_LDB_out = C_LDB_in + 4 * k

        # Transition Layer 輸出通道數 (壓縮後的通道數，保持 n*2 = 32)
        C_TL_out = C0

        # 將 LDB 和 TransLayer 組合成一個 Sequential 模塊
        block = nn.Sequential(
            LDB(C_LDB_in, t),
            TransLayer(C_LDB_out, C_TL_out),
        )
        self.dense_blocks.append(block)
        # 更新下一輪的輸入通道數
        current_channels = C_TL_out

    self.layer3 = nn.Sequential(
        nn.AdaptiveAvgPool2d((7,7)),
        nn.Flatten(),
        nn.Linear(C0, 128),
        nn.ReLU(),
        nn.Linear(128, num_class)
    )


  def forward(self, x):
      x = self.layer1(x)

      for block in self.dense_blocks:
          # LDB: (N, 32, H, W) -> (N, 96, H, W)
          # TransLayer: (N, 96, H, W) -> (N, 32, H, W)
          x = block(x)

      x = self.layer3(x)

      return x



