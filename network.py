import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import torch.nn.init as init


class StandardConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(StandardConv1d, self).__init__()
        self.StandardConv1d = nn.Sequential(
            nn.Conv1d(in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = stride //2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.StandardConv1d(x)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.DepthwiseSeparableConv1d = nn.Sequential(
            nn.Conv1d(in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels = out_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = kernel_size // 2,
                    groups = out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace = True))

    def forward (self, x):
        return self.DepthwiseSeparableConv1d(x)

class GRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional):
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True, bidirectional=bidirectional)
        
        self.conv = nn.Sequential(nn.Conv1d(hidden_size * (2 if bidirectional==True else 1), out_channels, kernel_size = 1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        output,h = self.GRU(x)
        output = output.transpose(1,2)
        output = self.conv(output)
        return output

class FirstTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(FirstTrCNN, self).__init__()
        self.FirstTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = stride//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.FirstTrCNN(x)


class TrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TrCNN, self).__init__()
        self.TrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = stride//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1,x2),1)
        output = self.TrCNN(x)
        return output

class LastTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(LastTrCNN, self).__init__()
        self.LastTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding=stride//2))

    def forward(self,x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1,x2),1)
        output = self.LastTrCNN(x)
        return output

class TRUNet(nn.Module):
    def __init__(self,
                input_size,
                channels_input,
                channels_output,
                channels_hidden,
                kernel_sizes,
                strides,
                tr_channels_input
    ):
        super(TRUNet, self).__init__()
        #Encoder
        self.down1 = StandardConv1d(input_size, channels_input , kernel_sizes[0], strides[0])
        self.down2 = DepthwiseSeparableConv1d(channels_input, channels_hidden, kernel_sizes[1], strides[1])
        self.down3 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[0], strides[0])
        self.down4 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[1], strides[1])
        self.down5 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[0], strides[0])
        self.down6 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[1], strides[0])
        
        #Bottleneck
        self.FGRU = GRUBlock(channels_hidden, channels_hidden//2, channels_hidden//2, bidirectional=True)
        self.TGRU = GRUBlock(channels_hidden//2, channels_hidden, channels_hidden//2, bidirectional=False)
        
        #Decoder
        self.up1 = FirstTrCNN(channels_hidden//2, channels_hidden//2, kernel_sizes[1], strides[0])
        self.up2 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[0], strides[0])
        self.up3 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[1], strides[1])
        self.up4 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[0], strides[0])
        self.up5 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[1], strides[1])
        self.up6 = LastTrCNN(channels_hidden, channels_output, kernel_sizes[0], strides[0])
  

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        
        x7 = x6.transpose(1,2)
        x8 = self.FGRU(x7)
        
        x9 = x8.transpose(1,2)
        x10 = self.TGRU(x9)
        
        x11 = self.up1(x10)
        x12 = self.up2(x11,x5)
        x13 = self.up3(x12,x4)
        x14 = self.up4(x13,x3)
        x15 = self.up5(x14,x2)
        x16 = self.up6(x15,x1)
        return x16
