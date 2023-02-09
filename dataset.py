import os
import numpy as np

from scipy.io.wavfile import read as wavread
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio import transforms

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms
import torchaudio


class PCENTransform(nn.Module):
  
  '''   code adopted from: 
        https://www.kaggle.com/code/simongrest/trainable-pcen-frontend-in-pytorch
        written by Simon Grest
        
        Computes trainable Per-Channel Energy Normalization (PCEN)
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
    x = x.transpose(2, 1)
    
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
    
    x = x.transpose(2, 1)
    return x




class DataPreprocessing:

    '''
    Data preprocessing class converts time-domain audio
    signal into structure of shape (timeframe, 4, freq_bins)
    where 4 dimension represents:
        (log_spctrogram,
         pcen transformed spectrogram,
         real part of demodulated phase,
         imag part of demodulated phase)
    
    Input:
        Tuple: ((1, time-domain signal) clean),
                (1, time-domain signal) noisy))
    
    Returns:
        Tuple: ((time-frame, 4, freq_bins) clean,
                (time-frame, 4, freq_bins) noisy)
    
    
    '''
    def __init__(self):
        self.pcen = PCENTransform()
        self.n_fft = 512
        self.hop_length = 128


    def _demod_phase(self, magnitude, phase):
        
        '''
        Calculates demodulated phase of real and imaginary

        Args:
            magnitude: (float32)
            phase:     (float32)

        Returns:
            real_demod (float32):   Demodulated phase of real
            imag_demod (float32):   Demodulated phase of imaginary
        '''
        
        #get carrier signal for real and imaginary part
        carrier_real = torch.cos(phase)
        carrier_imag = torch.sin(phase)
        
        #calculate real and imag demod
        real_demod = magnitude * carrier_real
        imag_demod = magnitude * carrier_imag

        return real_demod, imag_demod
    
    
    
    def _stft(self, audio, pcen=False):

        '''
        Compute complex form short-time fourier transform
        '''
        spectrogram = torch.stft(audio, n_fft = self.n_fft, hop_length = self.hop_length, return_complex = True)
        return spectrogram
            
    

    def _pre_pcen(self, mag, phase):
        return torch.sqrt(torch.clamp(mag**2 + phase**2, min=1e-7))

    
    def perm(self, tensor):
      '''
      permute function
      '''
      return tensor.permute(1, 0, 2)
   
   
    def __call__(self, audio):
        #get spectrogram from audio
        spec = self._stft(audio)
        
        #magnitude and phase
        mag, phase = torch.abs(spec), torch.angle(spec)

        #real and imag demodulated phase
        real_demod, imag_demod = self._demod_phase(mag, phase)
        
        #power to db conversion (log-spectrogram)
        log_spec = torch.log(mag)

        #apply pcen
        pcen = self.pcen(self._pre_pcen(mag, phase))
        
        #concatenate and permute data to be fed to the network
        data = torch.cat((self.perm(log_spec),
                          self.perm(pcen),
                          self.perm(real_demod),
                          self.perm(imag_demod)), dim = 1)
        
        #return data (time_frame, 4 features, freq_bins)
        return data



class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean_spectrogam, noisy_spectrogram, clean waveform, noisy waveform, file_id)
    
    Returns:
       (Clean Spectrogram, Noisy Spectrogram, Clean Audio, Noisy Audio, Filed id)
    """
    
    def __init__(self, root='./', subset='training', crop_length_sec=0):
        super(CleanNoisyPairDataset).__init__()
        

        self.dp = DataPreprocessing()
        
        assert subset is None or subset in ["training", "testing"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset

        N_clean = len(os.listdir(os.path.join(root, 'training_set/clean')))
        N_noisy = len(os.listdir(os.path.join(root, 'training_set/noisy')))
        assert N_clean == N_noisy

        if subset == "training":
            self.files = [(os.path.join(root, 'training_set/clean', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'training_set/noisy', 'fileid_{}.wav'.format(i))) for i in range(N_clean)]
        
        elif subset == "testing":
            sortkey = lambda name: '_'.join(name.split('_')[-2:])  # specific for dns due to test sample names
            _p = os.path.join(root, 'datasets/test_set/synthetic/no_reverb')  # path for DNS
            
            clean_files = os.listdir(os.path.join(_p, 'clean'))
            noisy_files = os.listdir(os.path.join(_p, 'noisy'))
            
            clean_files.sort(key=sortkey)
            noisy_files.sort(key=sortkey)

            self.files = []
            for _c, _n in zip(clean_files, noisy_files):
                assert sortkey(_c) == sortkey(_n)
                self.files.append((os.path.join(_p, 'clean', _c), 
                                   os.path.join(_p, 'noisy', _n)))
            self.crop_length_sec = 0

        else:
            raise NotImplementedError 

    

    def __getitem(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0])
        noisy_audio, sample_rate = torchaudio.load(fileid[1])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0) #check if this applicable for spectrogram conversion
        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)
            
        #random crop in the time domain
        if self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]

        #prepare audio signal and spectrogram data pairs
        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)
        clean_features, noisy_features = self.dp(clean_audio), self.dp(noisy_audio)
        
        #make input shape suitable for network
        return (clean_features, noisy_features, clean_audio, noisy_audio, fileid)

    
    def __len__(self):
        return len(self.files)


def load_CleanNoisyPairDataset(root,
                               subset,
                               crop_length_sec,
                               batch_size,
                               sample_rate,
                               num_gpus = 1):
        
        dataset = CleanNoisyPairDataset(root = root, subset = subset, crop_length_sec=crop_length_sec)
        kwargs = {'batch_size': batch_size,
                  'num_workers': 4,
                  'pin_memory': False,
                  'drop_last': False}
        
        if num_gpus > 1:
            train_sampler = DistributedSampler(dataset)
            dataloader = torch.utils.data.Dataloader(dataset, sampler=train_sampler, **kwargs)
        else:
            dataloader = torch.utils.data.Dataloader(dataset, sampler=None, shuffle=True, **kwargs)
        
        return dataloader


if __name__ == '__main__':
    import json
    with open('./configs/tiny.json') as f:
        data = f.read()
    config = json.loads(data)
    trainset_config = config('train')

    trainloader = load_CleanNoisyPairDataset(**trainset_config,
                                             subset='training',
                                             batch_size=2,
                                             num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config,
                                            subset='testing',
                                            batch_size=2,
                                            num_gpus=1)
    print(len(trainloader), len(testloader))

    for clean_feat, noisy_feat, clean_audio, noisy_audio, fileid in trainloader:
        
        clean_feat = clean_feat.cuda()
        noisy_feat = noisy_feat.cuda()
        
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        
        print(clean_feat.shape, noisy_audio.shape, fileid)
        print(clean_audio.shape, noisy_audio.shape, fileid)
        break       
