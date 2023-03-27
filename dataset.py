import os
import numpy as np
import librosa

from scipy.io.wavfile import read as wavread
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio import transforms

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

#Calcualate Per-Channel Energy Normalization
def pcenfunc(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
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


class ProcessAudio(nn.Module):

  def __init__(self,
               n_fft = 512,
               hop_length = 128,
               sample_rate = 16000,
               min_level_db = -100):
    
    super().__init__()
    self.n_fft = n_fft
    self.n_mels = self.n_fft // 2 + 1
    self.hop_length = hop_length
    self.sample_rate = sample_rate
    self.min_level_db = min_level_db
    self.sr = 48000
    self.target_sr = 16000
    self.spec = T.Spectrogram(n_fft = self.n_fft, 
                              hop_length = self.hop_length, 
                              power=None, 
                              normalized=False)
    
    self.inv_spec = T.InverseSpectrogram(n_fft = self.n_fft, 
                                      hop_length = self.hop_length, 
                                      normalized=False)

  def demod_phase(self, phase):
      
      '''
      Calculates demodulated phase of real and imaginary
      Args:
          spectrogram
      Returns:
          real_demod (float32):   Demodulated phase of real
          imag_demod (float32):   Demodulated phase of imaginary
      '''
      phase = phase.squeeze(0).cpu().numpy()
      
      #calculate demodulated phase
      demodulated_phase = np.unwrap(phase)
      demodulated_phase = torch.from_numpy(demodulated_phase).unsqueeze(0).cuda()
      
      #get real and imagniary parts of the demodulated phase
      real_demod = torch.sin(demodulated_phase)
      imag_demod = torch.cos(demodulated_phase)

      return real_demod, imag_demod
  
  
  def mod_phase(self, magnitude, real_demod, imag_demod):
      """
      Reverse operation of demodulation
      Args:
        real_demod(float32): real part of the demodulated signal
        imag_demod(float32): imaginary part of the demodulated signal
      Returns:
        Spectrogram(comeplx64)
      
      """
      #wrap phase back its original state 
      wrap = torch.arctan2(real_demod, imag_demod)

      #construct complex spectrogram
      complex_spectrogram = torch.exp(magnitude) * torch.exp(1j * wrap)
      return complex_spectrogram.unsqueeze(0)


  def log_mag(self, magnitude):
      return torch.log(magnitude + 1e-9)
  
  
  def get_mag_phase(self, spectrogram):
    magnitude = torch.abs(spectrogram)
    phase = torch.angle(spectrogram)
    return magnitude, phase

  
  def istft(self, complex_spec):
    return self.inv_spec(complex_spec)
  
  
  def perm(self, tensor):
      return tensor.permute(2, 0, 1)

  def de_perm(self, tensor):
    return tensor.permute(1, 2, 0)
  
  
  def norm(self, audio):
    mean = torch.mean(audio, dim=1)
    std = torch.std(audio, dim=1)
    return (audio - mean / std)

  
  def forward(self, audio):
    
    audio = self.norm(audio)
    spectrogram = self.spec(audio)
    magnitude, phase = self.get_mag_phase(spectrogram)
    real_demod, imag_demod = self.demod_phase(phase)
    features = torch.cat((self.perm(self.log_mag(magnitude)),
                          self.perm(real_demod),
                          self.perm(imag_demod)), dim=1)  
    return features



class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean_spectrogam, noisy_spectrogram, clean waveform, noisy waveform, file_id)
    
    Returns:
       (Clean Spectrogram, Noisy Spectrogram, Clean Audio, Noisy Audio, Filed id)
    """
    
    def __init__(self, root='./', 
                 subset='training', 
                 crop_length_sec=0):
        
        super(CleanNoisyPairDataset).__init__()        
        assert subset is None or subset in ["training", "testing"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset
        self.resampler = T.Resample(48000, 16000)
        
        N_clean = len(os.listdir(os.path.join(root, 'clean')))
        N_noisy = len(os.listdir(os.path.join(root, 'noisy')))
        assert N_clean == N_noisy

        if subset == "training":
            self.files = [(os.path.join(root, 'clean', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'noisy', 'fileid_{}.wav'.format(i))) for i in range(N_clean)]
        
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

    

    def __getitem__(self, n):
        
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0], normalize=True)
        noisy_audio, sample_rate = torchaudio.load(fileid[1], normalize=True)
        
        #resample from 48kHz to 16kHz
        
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio)
        #random crop audio
        
        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)
            
        #random crop in the time domain
        if self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]
            
            clean_audio = clean_audio.unsqueeze(0)
            noisy_audio = noisy_audio.unsqueeze(0)
        return (clean_audio, noisy_audio, fileid)


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
            dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        
        return dataloader


if __name__ == '__main__':
    import json
    with open('/content/tinyrecurrentunet/config/tiny.json') as f:
        data = f.read()
    config = json.loads(data)
    trainset_config = config['trainset']

    trainloader = load_CleanNoisyPairDataset(**trainset_config,
                                             subset='training',
                                             batch_size=2,
                                             num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config,
                                            subset='testing',
                                            batch_size=2,
                                            num_gpus=1)
    print(len(trainloader), len(testloader))

    for clean_audio, noisy_audio, fileid in trainloader:
        
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        
        print(clean_audio.shape, noisy_audio.shape, fileid)
        break 
