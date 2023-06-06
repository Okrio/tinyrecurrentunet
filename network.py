import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import *
from phm import PhaseAwareMask

#TRUNet with 1 dimensional convolutions
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
                tr_channels_input):
        
        super(TRUNet, self).__init__()
        
        self.encoder = nn.ModuleList(nn.Sequential(self.down1 = StandardConv1d(3, 64, 5, 2),
                                     DepthwiseSeparableConv1d(64, 128, 3, 1),
                                     DepthwiseSeparableConv1d(128, 128, 5, 2),
                                     DepthwiseSeparableConv1d(128, 128, 3, 1),
                                     DepthwiseSeparableConv1d(128, 128, 5, 2),
                                     DepthwiseSeparableConv1d(128, 128, 3, 2)))
        
        self.decoder = nn.ModuleList(FirstTrCNN(64, 64, 3, 2),
                                     TrCNN(192, 64, 5, 2),
                                     TrCNN(192, 64, 3, 1),
                                     TrCNN(192, 64, 5, 2),
                                     TrCNN(192, 64, 3, 1),
                                     LastTrCNN(128, 8, 5, 2))
        

        self.FGRU = GRUBlock(128, 64, 64, bidirectional=True)
        self.TGRU = GRUBlock(64, 128, 64, bidirectional=False)
        

    def forward(self, x):
        
        skip_connections = []
        
        for down_sample in self.encoder:
            x = down_sample(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]
        
        x = x.transpose(1, 2)
        x = self.FGRU(x)
        x = x.transpose(1, 2)
        
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x += skip_i[:, :, :x.shape[-1]]
            x = upsampling_block(x)

        return x

    
