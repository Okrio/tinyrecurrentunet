import torch
import torch.nn as nn
import torch.functional as F
import tqdm
from torchaudio.transforms import MelScale, Spectrogram


class GradInverse(nn.Module):
    
    def __init__(self):
        super(GradInverse, self).__init__()
        self.sc_coeff = 20
        self.transform_fn = Spectrogram(n_fft=512, win_length=512, hop_length=128)

    
    def spectral_convergence(self, input, target):
        return self.sc_coeff * ((input - target).norm().log10() - target.norm().log10())

    def Gradient(self, 
             spec, 
             samples = None, 
             init_x0 = None, 
             maxiter =1000, 
             tol = 1e-6, 
             verbose = 1, 
             evaiter = 10, 
             lr = 0.003, 
             hop = 128):

        spec = torch.Tensor(spec)
        samples = (spec.shape[-1]*hop)-hop

        if init_x0 is None:
            init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
        x = nn.Parameter(init_x0)
        T = spec

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam([x], lr=lr)

        bar_dict = {}
        metric_func = self.spectral_convergence
        bar_dict['spectral_convergence'] = 0
        metric = 'spectral_convergence'

        init_loss = None
        with tqdm(total=maxiter, disable=not verbose) as pbar:
            for i in range(maxiter):
                optimizer.zero_grad()
                V = self.transform_fn.forward(x)
                loss = criterion(V, T)
                loss.backward()
                optimizer.step()
                lr = lr * 0.9999
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                if i % evaiter == evaiter - 1:
                    with torch.no_grad():
                        V = self.transform_fn.forward(x)
                        bar_dict[metric] = metric_func(V, spec).item()
                        l2_loss = criterion(V, spec).item()
                        pbar.set_postfix(**bar_dict, loss=l2_loss)
                        pbar.update(evaiter)

        return x

    def forward(self, x):
        return self.gradient(x)
