import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

from scipy.io.wavfile import read as wavread

import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T
from torchaudio import functional as F
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio import transforms


import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def diff(x, axis):
    shape = x.shape
    begin_back = [0 for unused_s in range(len(shape))]
    begin_front = [0 for unused_s in range(len(shape))]
    begin_front[axis] = 1
    size = list(shape)
    size[axis] -= 1
    slice_front = x[begin_front[0]:begin_front[0] + size[0], begin_front[1]:begin_front[1] + size[1]]
    slice_back = x[begin_back[0]:begin_back[0]+size[0], begin_back[1]:begin_back[1]+size[1]]
    d = slice_front - slice_back
    return d


def unwrap(p, axis = -1):
  pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
  dd = diff(p, axis=axis)
  ddmod = torch.remainder(dd + pi, 2.0 * pi) - pi
  idx = torch.logical_and(torch.eq(ddmod, -pi), torch.greater(dd, 0))
  ddmod = torch.where(idx, torch.ones_like(ddmod) * pi, ddmod)
  ph_correct = ddmod - dd
  idx = torch.less(torch.abs(dd), pi)
  ddmod = torch.where(idx, torch.zeros_like(ddmod), dd)
  ph_cumsum = torch.cumsum(ph_correct, axis=axis)
  shape = torch.tensor(p.shape)
  shape[axis] = 1
  #ph_cumsum = torch.cat([torch.zeros(list(shape)), ph_cumsum], axis=axis)
  unwrapped = p + ph_cumsum
  return unwrapped.squeeze(0)



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


class DataAugment:   
    
    '''
    Applies data augmentation on the target tensor signal
    
    Augmentations:
        Gain:                              randomised between -12 and -5 dB
        Low-Pass bi-quad filter:           randomised between 7kHz and 10kHz
        Hi-Pass bi-quad filter:            randomised between 800Hz and 1.2kHz
    
    Returns:
        Processed tensor signal
 
    '''
    
    def __init__(self):

        #gain in dB
        self.min_gain = -12.0
        self.max_gain = -5.0

        #frequency in Hz
        self.lp_min = 7000
        self.lp_max = 10000

        self.hp_min = 800
        self.hp_max = 1200

        #samplin rate
        self.sr = 48000

        self.gains = torch.arange(self.min_gain, self.max_gain, 0.033)
        self.lp_freqs = torch.arange(self.lp_min, self.lp_max, 100)
        self.hp_freqs = torch.arange(self.hp_min, self.hp_max, 50)
    
    
    def __call__(self, x):
      
      #get audgmentation parameters
      lp_cutoff = random.choice(self.lp_freqs)
      hp_cutoff = random.choice(self.hp_freqs)
      gain = random.choice(self.gains)
      
      #apply augmentation
      x = F.gain(x, gain_db = gain)
      x = F.lowpass_biquad(x, self.sr, lp_cutoff, Q = 0.7)
      x = F.highpass_biquad(x, self.sr, hp_cutoff, Q = 0.7)
      return x


    
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
    self.min_level_db = -100.
    self.ref_level_db = 25.
    self.spec = T.Spectrogram(n_fft = self.n_fft, 
                              hop_length = self.hop_length, 
                              power=None, 
                              normalized=False)
    self.inv_spec = T.InverseSpectrogram(n_fft = self.n_fft, 
                                      hop_length = self.hop_length, 
                                      normalized=False)


  def get_mag_phase(self, spectrogram):
    magnitude = torch.abs(spectrogram).squeeze(0)
    phase = torch.angle(spectrogram)
    return magnitude, phase   
  
  
  def demod_phase(self, phase):
      
      '''
      Calculates demodulated phase of real and imaginary
      Args:
          Phase (float32):        Phase of the clean signal
      Returns:
          real_demod (float32):   Demodulated phase of the real part of the clean signal
          imag_demod (float32):   Demodulated phase of imaginary the part of the clean signal
      '''
      
      demodulated_phase = unwrap(phase)
      
      #get real and imagniary parts of the demodulated phase
      real_demod = torch.sin(demodulated_phase)
      imag_demod = torch.cos(demodulated_phase)

      return real_demod, imag_demod
  
  
  def mod_phase(self, magnitude, real_demod, imag_demod):
      """
      Reverse function of demodulation
      Args:
        magnitude(float32):  denoised magnitude spectrogram
        real_demod(float32): real part of the demodulated denoised signal
        imag_demod(float32): imaginary part of the demodulated denoised signal
      Returns:
        Spectrogram(comeplx64)
      
      """
      #reverse of unwrap function
      wrap = torch.arctan2(real_demod, imag_demod)
    
      #apply de-norm and dB to Amplitude on the denoised magnitude
      magnitude = (self.db_to_amp(self.de_norm(magnitude)))
      
      #construct complex spectrogram
      complex_spectrogram = magnitude * torch.exp(1j * wrap)
      #complex_spectrogram = magnitude * torch.exp(torch.complex(torch.zeros([1]), 
      #                                                          torch.ones([1])) * wrap)
      return complex_spectrogram.unsqueeze(0)


    
  def amp_to_db(self, magnitude):
      '''
      Amplitude to DB
      '''
      return 20 * torch.log10(torch.clamp(magnitude, min=1e-7)) - self.ref_level_db
     

  def db_to_amp(self, db_spec):
      """
      DB to Amplitude
      """
      return torch.pow(10, db_spec / 20.0)


  def perm(self, tensor):
      return tensor.permute(2, 0, 1)


  def de_perm(self, tensor):
    return tensor.permute(1, 2, 0) 
  
  
  def norm(self, db_spec):
        """
        normalize dB lavel spectrogram values to be
        scaled between [-1, 1] using external 
        minimum level
        """
        return torch.clamp((((db_spec - self.min_level_db) / -self.min_level_db)*2.)-1., -1, 1) 
  
  
  def de_norm(self, norm_spec):
        """
        de-normalize spectrogram values to dB level using 
        external minimum level
        """
        return (((torch.clamp(norm_spec, -1, 1) +1.) / 2.) * -self.min_level_db) + self.min_level_db + self.ref_level_db


  def forward(self, audio):
    """
    function to convert audio tenor to signal to feature tensor
    
    Argument:
        Signal (Tensor)
        shape: (1, time)
       
    Returns:
        Features (Tensor)
        shape: (751, 3, 257) 
    
    """
    #spectrogram = self.spec(audio)
    spectrogram = torch.stft(audio.squeeze(0),
                             n_fft = self.n_fft,
                             hop_length = self.hop_length,
                             normalized = False,
                             return_complex=True)
                          
    magnitude, phase = self.get_mag_phase(spectrogram.unsqueeze(0))
    real_demod, imag_demod = self.demod_phase(phase)
    features = torch.cat((self.perm(self.norm(self.amp_to_db(magnitude))),
                          self.perm(real_demod),
                          self.perm(imag_demod)), dim=1)
    
    return features


  def backward(self, denosied_features):
    """
    function to convert features tensors back to audio time-domain tensor
    
    Argument:
        features (Tensor)
        shape: (751, 3, 257) 
       
    Returns:
        Features (Tensor)
        shape: (1, time)  
    """
    
    denoised_mag, denoised_real, denoised_imag = self.de_perm(denoised_features)
    modulate_denoised = self.mod_phase(denoised_mag, 
                                      denoised_real, 
                                      denoised_imag)
    #denoised_audio = self.inv_spec(modulate_denoised)
    denoised_audio = torch.istft(modulate_denoised,
                                 n_fft = self.n_fft,
                                 hop_length = self.hop_length,
                                 normalized=False)
    
    return denoised_audio

    
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
        self.root = root
        self.aug = DataAugment()
        self.crop_length_sec = crop_length_sec
        self.subset = subset
        
        N_clean = len(os.listdir(os.path.join(root, 'clean')))
        #N_noisy = len(os.listdir(os.path.join(root, 'noisy')))
        #assert N_clean == N_noisy

        if subset == "training":
            self.files = [(os.path.join(root, 'clean', 'fileid_{}.wav'.format(i))) for i in range(N_clean)]
            
            self.noise_files = os.listdir(os.path.join(root, 'keyboard'))
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
        noise_file = random.choice(self.noise_files)
        noise_file = os.path.join(self.root, 'keyboard/'+ noise_file)
        
        clean_audio, sample_rate = torchaudio.load(fileid, normalize=True)
        noise_audio, sample_rate = torchaudio.load(noise_file, normalize=True)
        #noise_audio, sample_rate = torchaudio.load(fileid[1], normalize=True)
        
        
        #apply augmentation on noise
        noise_audio = self.aug(noise_audio)
        #noisy_audio = clean_audio + noise_audio
        
        #clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        clean_audio = clean_audio.squeeze(0)
        #assert len(clean_audio) == len(noisy_audio)
        #random crop audio
        
        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)
            
        #random crop in the time domain
        if self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            #noisy_audio = noisy_audio[start:(start + crop_length)]
            noisy_audio = clean_audio + noise_audio
            
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
    with open('/home/tinyrecurrentunet/config/tiny.json') as f:
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
