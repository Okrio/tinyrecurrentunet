import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import torch.nn.init as init

class PCENTransform(nn.Module):
  
  '''   code adopted from: 
        https://www.kaggle.com/code/simongrest/trainable-pcen-frontend-in-pytorch
        written by Simon Grest
        
        Computes trainable and static Per-Channel Energy Normalization (PCEN)
        to normalize the energy of each channel seperately.
        The PCEN process aims to enhance the dynamic range 
        of the audio, by increasing the relative amplitude 
        of lower energy signals, and decreasing the relative 
        amplitude of higher energy signals. This helps the neural-
        network to better handle variations in the audio signal 
        and improves the robustness of the model to different levels 
        of audio energy.
        
        It is recommended to apply PCEN before slicing the spectrograms;
        during the preprocessing as there seems to be some information
        losses when slicing is applied prior to PCEN transform. Therefore,
        we set the parameter "trainable" to False.
        
        Input:
            Magniture spectrogram of the signal converted to dB scale

        Args:
            eps   (float):
            s     (float):
            delta (int):
            r     (float)
            alpha (float):         Hyperparameter controlling the 'strength' of the Power-law compreession
            r     (float):         Hyperparameter controlling the 'shape' of the Power-law compreession
            trainable(str):        Boolean to make PCEN parameters trainable during training iterations
        
        Returns:
            Normalized Spectrogram
        
  '''
  def __init__(self,
               eps = 1e-6,
               s = 0.025,
               alpha = 0.98,
               delta = 2,
               r = 0.5,
               trainable = False):
    super().__init__()
    if trainable:
      self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))
      self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))
      self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))
      self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))
    
    else:
      self.s = s
      self.alpha = alpha
      self.delta = delta
      self.r = r
    
    self.eps = eps
    self.trainable = trainable

  
  def _pcen(self,
           x, 
           eps=1E-6, 
           s=0.025, 
           alpha=0.98, 
           delta=2, 
           r=0.5, 
           training=False):
    
    frames = x.split(1, -2)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = s * frame
            m_frames.append(last_state)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_

  
  def forward(self, x):
    x = x.permute((0, 1, 3, 2)).squeeze(dim=1)
    
    if self.trainable:
      x = self._pcen(x, self.eps, torch.exp(self.log_s), 
                                  torch.exp(self.log_alpha), 
                                  torch.exp(self.log_delta), 
                                  torch.exp(self.log_r), 
                                  self.training and self.trainable)
    else:
      x = self._pcen(x, self.eps,
                        self.s,
                        self.alpha,
                        self.delta,
                        self.r,
                        self.training and self.trainable)
    
    x = x.unsqueeze(dim=1).permute((0, 1, 3, 2))
    return x




class PhaseAwareMask(nn.Module):
  
  def __init__(self, 
               beta,
               fft_size,
               hop_size):
       
    """
    Input:
      Mixture (audio tensor):   audio signal containing speech and noise
      Estimated (spectrogram):  denoised output from TRUNet


    Computes the Phase-aware β-sigmoid mask using magnitude and phase 
    information of mixture and estimated signal.

    Args:
      beta: mask coefficent controlling the sharpness of the mask - torch
      mixture (complex): audio signal containing speech and noise combined 
      estimated (float): output of the network. Estimated denoised audio signal

    Returns:
      Masked Spectrogram of Shape:
    
    """

    super(PhaseAwareMask, self).__init__()
    self.beta = beta
    self.fft_size = self.fft_size
    self.hop_size = self.hop_size
    self.return_complex = True

  
  def _stft(self, x, 
            fft_size, 
            hop_size, 
            win_length=None, 
            window=None, 
            return_complex=True):
    
    x_stft = torch.stft(x, 
                        fft_size, 
                        hop_size, 
                        win_length, 
                        window, 
                        return_complex = return_complex)
    return x_stft  
 
  
  def forward(self, mixture, estimated):

    #compute stft
    stft_mixture = self._stft(mixture, fft_size = self.fft_size, 
                              hop_size = self.hop_size, 
                              return_complex = self.return_complex)
    
    #extract the magnitude and phase of the stft
    mag_mix = torch.abs(stft_mixture)
    phase_mix = torch.angle(stft_mixture)

    #extract the phase of the estimated sources 
    phase_est = torch.angle(estimated)

    #compute the soft mask
    soft_mask = 1 / (1 + torch.exp(-self.beta * (phase_mix - phase_est)))

    #apply the soft mask to the magnitude of the mixture
    estimated = soft_mask * mag_mix
    return estimated




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
    def forward(self,x):
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

    def forward(self,x1,x2):
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
        Model class compiles the TRUNet into an instance.
        
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
        x1 = torch.reshape(x, (x.shape[1]//self.input_size, self.input_size, x.shape[2]))
        
        #PCEN
        
        ######
        #here#
        ######
        
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
        y =   self.up6(x15, x1)

        #reverts back to original shape
        y = torch.reshape(y, (-1, y.shape[0] * y.shape[1], T))
        
        #Phase-aware β-sigmoid mask
        
        ######
        #here#
        ######
        
        
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



if __name__=='__main__':
    
    import json
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/content/tiny.json', 
          help='Json Configuration file')
    args = parser.parse_args()
    
    #load arguments from json file
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    network_config = config["network"]
    
    #load model
    TRU = TRUNet(**network_config).cuda
    total_params = sum(p.numel() for p in TRU.parameters())
    
    print("total TRUNet params:", total_params)
