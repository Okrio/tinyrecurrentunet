import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import *



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
        self.down1 = StandardConv1d(4,64,5,2)
        self.down2 = DepthwiseSeparableConv1d(64, 128, 3, 1)
        self.down3 = DepthwiseSeparableConv1d(128, 128, 5, 2)
        self.down4 = DepthwiseSeparableConv1d(128, 128, 3, 1)
        self.down5 = DepthwiseSeparableConv1d(128, 128, 5, 2)
        self.down6 = DepthwiseSeparableConv1d(128, 128, 3, 2)
        self.FGRU = GRUBlock(128, 64, 64, bidirectional=True)
        self.TGRU = GRUBlock(64, 128, 64, bidirectional=False)
        self.up1 = FirstTrCNN(64, 64, 3, 2)
        self.up2 = TrCNN(192, 64, 5, 2)
        self.up3 = TrCNN(192, 64, 3, 1)
        self.up4 = TrCNN(192, 64, 5, 2)
        self.up5 = TrCNN(192, 64, 3, 1)
        self.up6 = LastTrCNN(128, 4, 5, 2)
  

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

    


def pointwise(in_channels, out_channels):
    return Sequential(
        Conv2d(in_channels, out_channels, 1, 1),
        BatchNorm2d(out_channels),
        ReLU(),
    )


def depthwise(in_channels, out_channels, kernel_size, stride):
    return Sequential(
        Conv2d(in_channels, 
               out_channels, 
               (kernel_size, 1), 
               (stride, 1), 
                padding=(kernel_size // 2, 0), groups=in_channels),
        BatchNorm2d(out_channels),
        ReLU(),
    )


class TRUNet2D(Module):
    def __init__(self, in_channels=4, out_channels=10):
        super().__init__()
        self.encoder = ModuleList([
            Sequential(Conv2d(in_channels, 64, (5, 1), (2, 1), padding=(2, 0)), BatchNorm2d(64), ReLU()),
            Sequential(pointwise( 64, 128), depthwise(128, 128, 3, 1)),
            Sequential(pointwise(128, 128), depthwise(128, 128, 5, 2)),
            Sequential(pointwise(128, 128), depthwise(128, 128, 3, 1)),
            Sequential(pointwise(128, 128), depthwise(128, 128, 5, 2)),
            Sequential(pointwise(128, 128), depthwise(128, 128, 3, 2)),
        ])
        self.fgru = Sequential(
            GRU(128, 64, bidirectional=True, batch_first=True),
            pointwise(128, 64),
        )
        self.tgru = ModuleList([
            GRU(64, 128, batch_first=True),
            Linear(128, 64),
            Sequential(BatchNorm2d(64), ReLU()),
        ])
        self.decoder = Sequential(
            Sequential(pointwise(64, 64), ConvTranspose2d(64, 64, (3, 1), (2, 1), padding=(1, 0), output_padding=(1, 0))),
            Sequential(pointwise(192, 64), ConvTranspose2d(64, 64, (5, 1), (2, 1), padding=(2, 0), output_padding=(1, 0))),
            Sequential(pointwise(192, 64), ConvTranspose2d(64, 64, (3, 1), (1, 1), padding=(1, 0))),
            Sequential(pointwise(192, 64), ConvTranspose2d(64, 64, (5, 1), (2, 1), padding=(2, 0), output_padding=(1, 0))),
            Sequential(pointwise(192, 64), ConvTranspose2d(64, 64, (3, 1), (1, 1), padding=(1, 0))),
            Sequential(pointwise(128, out_channels), ConvTranspose2d(out_channels, out_channels, (5, 1), (2, 1), padding=(2, 0), output_padding=(1, 0))),
        )

    def forward(self, x: "(B, in_channels, 256, T)"):
        batch, _, freqs, time = x.shape
        if freqs == 257:
            x = x[:, :, :256]

        # Encoder
        encoder_outs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outs.append(x)

        # FGRU block
        fgru_gru, fgru_pointwise = self.fgru
        fgru_gru_in = x.permute(0, 3, 2, 1).flatten(0, 1)
        fgru_gru_out, fgru_gru_state = fgru_gru(fgru_gru_in)
        fgru_pointwise_in = fgru_gru_out.reshape(batch, time, 16, 128).permute(0, 3, 2, 1)
        fgru_pointwise_out = fgru_pointwise(fgru_pointwise_in)

        # TGRU block
        tgru_gru, tgru_linear, tgru_bnact = self.tgru
        tgru_gru_in = fgru_pointwise_out.permute(0, 2, 3, 1).flatten(0, 1)
        tgru_gru_out, tgru_gru_state = tgru_gru(tgru_gru_in)
        tgru_linear_in = tgru_gru_out.reshape(batch, 16, time, 128)
        tgru_linear_out = tgru_linear(tgru_linear_in)
        tgru_bnact_in = tgru_linear_out.permute(0, 3, 1, 2)
        tgru_bnact_out = tgru_bnact(tgru_bnact_in)

        # Decoder
        x = tgru_bnact_out
        for i, (layer, skip_conn) in enumerate(zip(self.decoder, encoder_outs[::-1])):
            if i:
                x = torch.cat([x, skip_conn], dim=1)
            x = layer(x)

        if freqs == 257:
            x = functional.pad(x, [0, 0, 0, 1])
        return x
