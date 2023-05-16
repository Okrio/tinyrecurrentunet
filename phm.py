#Phase-aware β-sigmoid

import torch
import torch.nn as nn


class PhaseAwareMask(nn.Module):
  
  def __init__(self, 
               beta = 0.5):
       
    """
    Computes the Phase-aware β-sigmoid mask using magnitude and phase 
    information of mixture and estimated signal.

    Input:
        Mixture (spectrogram):    spectrogram containing speech and noise
        Estimated (spectrogram):  noise spectrogram output from network

    Args:
        beta (float): mask coefficent controlling the sharpness of the mask
    
    Returns:
        Masked Spectrogram
        
    """

    super(PhaseAwareMask, self).__init__()
    self.beta = beta 
  
  def forward(self, mixture, estimated):

    #extract the magnitude and phase of the stft
    mag_mixture = torch.abs(mixture)
    phase_mixture = torch.angle(mixture)

    #extract the phase of the estimated sources 
    phase_estimated = torch.angle(estimated)

    #compute the soft mask
    soft_mask = 1 / (1 + torch.exp(-self.beta * (phase_mix - phase_est)))

    #apply the soft mask to the magnitude of the mixture
    estimated = soft_mask * mag_mixture
    return estimated

