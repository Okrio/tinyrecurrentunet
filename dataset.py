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
        self.n_fft = 512
        self.n_mels = self.n_fft // 2 + 1
        self.hop_length = 128
        self.sample_rate = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, 
                                                        n_mels = self.n_mels, 
                                                        n_fft = self.n_fft, 
                                                        hop_length = self.hop_length)
        
        self.atob = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        #self.pcen = pcen  
    
    
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
        return complex_spectrogram



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

        return real_demod, imag_demod


    def log_mag(self, spectrogram):
        x = torch.log(torch.abs(spectrogram)+1e-9)
        return x
    
    
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
            


    def _pcen(self, signal):
        #construct spectrogram complex --> float32
        mel = self.mel(signal)
        amp_to_db = self.atob(mel)
        to_pcen = pcenfunc(mel.permute((0,2,1)))
        return to_pcen.permute((0, 2, 1))

    
    def perm(self, tensor):
        '''
        permute function
        '''
        return tensor.permute(2, 0, 1)
   
    
    def normalise(self, audio):
        audio = audio.squeeze(0)
        mean = torch.mean(audio)
        std = torch.std(audio)
        norm_audio = (audio - mean) / std
        return norm_audio.unsqueeze(0)
   
    
    def __call__(self, audio):

        #get spectrogram from audio
        spectrogram = self._stft(audio)

        #calculate log-magnitude, real and imaginary part of demodulcated phase
        log_magnitude = self.log_mag(spectrogram)
        real_demod, imag_demod = self._demod_phase(spectrogram)

        #calculate PCEN
        pcen = self._pcen(audio)
        
        #concatenate and permute data to be fed to the network 
        data = torch.cat((self.perm(log_magnitude),
                          self.perm(pcen),
                          self.perm(real_demod),
                          self.perm(imag_demod)), dim = 1)
        
        
        #returns data of structure (time_frame, 4 features, freq_bins)
        return torch.nn.functional.normalize(data, dim=2)
