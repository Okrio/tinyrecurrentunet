import os
import numpy as np

from scipy.io.wavfile import read as wavread
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torchaudio.transforms as T

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms
import torchaudio



class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean waveform, noisy waveform, file_id)
    
    Returns:
       (Clean Spectrogram, Noisy Spectrogram, Filed id)
    """
    
    def __init__(self, root='./', subset='training', crop_length_sec=0):
        super(CleanNoisyPairDataset).__init__()
        
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

        #STFT parameters

    def _stft(self, x):
        x_stft = torch.stft(x, n_fft = 512, hop_length = 128, win_length = None, window = None)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1) #beware of the transpose - flipping the shape


    def __getitem(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0])
        noisy_audio, sample_rate = torchaudio.load(fileid[1])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0) #check if this applicable for spectrogram conversion
        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)
            
        #random crop
        if self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]

        #test
        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)
        clean_spec = self._stft(clean_audio)
        noisy_spec = self._stft(noisy_audio)

        #make input shape suitable for network
        return (clean_spec, clean_audio, clean_audio, noisy_audio, fileid)

    
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

    for clean_spec, noisy_spec, clean_audio, noisy_audio, fileid in trainloader:
        
        clean_spec = clean_spec.cuda()
        noisy_spec = noisy_spec.cuda()
        
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        
        print(clean_spec.shape, noisy_spec.shape, fileid)
        break       