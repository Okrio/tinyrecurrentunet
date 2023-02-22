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
import pcen


class DataProcessing:

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
        self.pcen = pcen.StreamingPCENTransform(n_mels=257, 
                                                n_fft=512, 
                                                hop_length=128, 
                                                trainable=True)
        self.n_fft = 512
        self.hop_length = 128
        self.n_mels = 257

    
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



    def _demod_phase(self, spectrogram):
        
        '''
        Calculates demodulated phase of real and imaginary

        Args:
            spectrogram

        Returns:
            real_demod (float32):   Demodulated phase of real
            imag_demod (float32):   Demodulated phase of imaginary
        '''

        phase = torch.angle(spectrogram)
        phase = phase.squeeze(0).detach().numpy() #to numpy 
        
        #calculate demodulated phase
        demodulated_phase = np.unwrap(phase)
        demodulated_phase = torch.from_numpy(demodulated_phase).unsqueeze(0)
        
        #get real and imagniary parts of the demodulated phase
        real_demod = torch.sin(demodulated_phase)
        imag_demod = torch.cos(demodulated_phase)

        return real_demod.requires_grad_(True), imag_demod.requires_grad_(True)


    def log_mag(self, spectrogram):
        x = torch.log(torch.abs(spectrogram))
        return torch.tensor(x, requires_grad=True)
    
    
    
    def istft(self, spectrogram):
        """
        Compute inverse short-time fourier transform
        """
        return torch.istft(spectrogram,
                             n_fft=self.n_fft,
                             hop_length=self.hop_length)

    
    def _stft(self, signal):

        '''
        Compute complex form short-time fourier transform
        '''
        return torch.stft(signal, 
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length,
                            return_complex=True)

    def normalise(self, data):
        return torch.nn.functional.normalize(data, 
                                             p=2.0, 
                                             dim=data.shape[-1], 
                                             eps=1e-12)   

    def _pcen(self, signal):
        x = self.pcen(signal)
        return x

    
    def perm(self, tensor):
        '''
        permute function
        '''
        return tensor.permute(2, 0, 1)
   
   
    
    def __call__(self, audio):
        
        #get spectrogram from audio
        spectrogram = self._stft(audio)

        #calculate log-magnitude, real and imaginary part of demodulcated phase
        log_magnitude = self.log_mag(spectrogram)
        real_demod, imag_demod = self._demod_phase(spectrogram)

        #calculate PCEN
        pcen = self._pcen(audio).permute(0, 2, 1)
        
        #concatenate and permute data to be fed to the network
        data = torch.cat((self.perm(log_magnitude.requires_grad_(False)),
                          self.perm(pcen).detach(),
                          self.perm(real_demod.requires_grad_(False)),
                          self.perm(imag_demod.requires_grad_(False))), dim = 1)
       
        #normalaise data
        data = self.normalise(data)
        
        #returns data of structure (time_frame, 4 features, freq_bins)
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
        

        self.dp = DataProcessing()
        
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

    

    def __getitem__(self, n):
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
        clean_features = self.dp(clean_audio)
        noisy_features =  self.dp(noisy_audio)
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
            dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        
        return dataloader


if __name__ == '__main__':
    import json
    with open('/content/tinyrecurrentunet/config/tiny.json') as f:
        data = f.read()
    config = json.loads(data)
    print('everrrrrrrr')
    trainset_config = config('trainset')
    
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

