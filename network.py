import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import torch.nn.init as init


class StandardConv1d(nn.Module):
   
    """
    Computes a standard 1-dimensional convolution
    """
   
    def __init__(self, in_channels, 
                        out_channels, 
                        kernel_size, 
                        stride):
        
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
    """
    Computes a depthwise separable 
    1-dimensional convolution
    """
    
    def __init__(self, in_channels, 
                       out_channels, 
                       kernel_size, 
                       stride):
                       
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
    def __init__(self, in_channels, 
                       hidden_size, 
                       out_channels, 
                       bidirectional):
        
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(in_channels, 
                          hidden_size, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        
        self.conv = nn.Sequential(nn.Conv1d(hidden_size * (2 if bidirectional==True else 1), 
                                            out_channels, 
                                            kernel_size = 1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        out, hidden = self.GRU(x)
        out = out.transpose(1,2)
        out = self.conv(out)
        return out


class FirstTrCNN(nn.Module):
    def __init__(self, in_channels, 
                       out_channels, 
                       kernel_size, 
                       stride):
        
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
    def forward(self, x):
        return self.FirstTrCNN(x)


class TrCNN(nn.Module):
    def __init__(self, in_channels, 
                       out_channels, 
                       kernel_size, 
                       stride):
        
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

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1, x2), 1)
        x = self.TrCNN(x)
        return x



class LastTrCNN(nn.Module):
    def __init__(self, in_channels, 
                       out_channels, 
                       kernel_size, 
                       stride):
        
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

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1, x2), 1)
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


        """
        TRUNet Class
        
        Args:
            input_size (int):                      input to the network
            channels_input (int):                  input Channels
            channels_output (int):                 output Channels
            channels_hidden (int):                 hidden gru channels
            tr_channels (int):                     input channels to TRCNN
            kernel_sizes (list of int):            kernel size
            strides (list of int):                 strides
        
        Returns:
            Denoised spectrogram
        """
       
        super(TRUNet, self).__init__()
        self.input_size = input_size
        self.down1 = StandardConv1d(input_size, channels_input, kernel_sizes[0], strides[0])
        self.down2 = DepthwiseSeparableConv1d(channels_input, channels_hidden,  kernel_sizes[1], strides[1])
        self.down3 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[0], strides[0])
        self.down4 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[1], strides[1])
        self.down5 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[0], strides[0])
        self.down6 = DepthwiseSeparableConv1d(channels_hidden, channels_hidden, kernel_sizes[1], strides[1])
        
        self.FGRU = GRUBlock(channels_hidden, channels_hidden//2, channels_hidden//2, bidirectional=True)
        self.TGRU = GRUBlock(channels_hidden//2, channels_hidden, channels_hidden//2, bidirectional=False)
        
        self.up1 = FirstTrCNN(channels_hidden//2, channels_hidden//2, kernel_sizes[0], strides[0])
        self.up2 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[0], strides[0])
        self.up3 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[1], strides[1])
        self.up4 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[0], strides[0])
        self.up5 = TrCNN(tr_channels_input, channels_hidden//2, kernel_sizes[1], strides[1])
        self.up6 = LastTrCNN(channels_hidden, channels_output, kernel_sizes[0], strides[0])
        
        self.initialize_weights()

    
    def forward(self, x):       
        # reshape incoming spectrogram
        #x = torch.reshape(x, (x.shape[1]//self.input_size, self.input_size, x.shape[2]))
        
        #Downsample
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        #Bottleneck
        x7 = x6.transpose(1, 2)
        x8 = self.FGRU(x7)
        x9 = x8.transpose(1, 2)
        
        #Upsample
        x10 = self.TGRU(x9)
        x11 = self.up1(x10)
        x12 = self.up2(x11,x5)
        x13 = self.up3(x12,x4)
        x14 = self.up4(x13,x3)
        x15 = self.up5(x14,x2)
        y =   self.up6(x15,x1)
       
        #reverts back to original shape
        #y = torch.reshape(y, (-1, y.shape[0] * y.shape[1], y.shape[2]))
        return y
    
    #weight initialization using normal distrbution
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.normal_(m.weight)           
            
            elif isinstance(m, nn.ConvTranspose1d):
                init.normal_(m.weight)   

            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
