import torch
import torch.nn as nn

class CosSimLoss(nn.Module):
    """
    Cosine Similary Loss class

    Args: 
        eps (float): Small value to avoid division by zero in cosine similarity caluculation
        g (list):    Segments length 
    
    Call:    
        x: Tensor (input audio signal)
        y: Tensor (target audio signal)
    
    
    Returns:
        Tensor (average sum of cosine similary of a input and 
                target tensors sliced with various lengths)
    """
    
    def __init__(self,
                eps = 1e-5,
                g = [4062, 2032, 1016, 508]):
        
        super(CosSimLoss, self).__init__()
        self.eps = eps
        self.g = g
        self.m = len(self.g)

    def cos_sim_func(self, input, target):
        '''
        Computes cosine similarity of input and target
        '''
        cos = nn.CosineSimilarity(dim=1, eps = self.eps)
        cos_sim_loss = 1 - cos(input, target)
        return cos_sim_loss
    
    
    def forward(self, x, y):
        loss = []
        for i in range(len(self.g)):     
          
          #get segment N/Mj and slice audio signal
          seg = self.g[i]
          x = x[:, :seg]
          y = y[:, :seg]
          
          #calculate cosine similarity function
          c = self.cos_sim_func(x, y)
          loss.append(c)
       
        return torch.sum(1 / self.m * torch.sum(torch.tensor(loss)))

