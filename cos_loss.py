class CosSimLoss(nn.Module):
    """
    Calculates Cosine Similary Loss for specific slices of
    input and target signals, namely the result from the model
    and the ground truth (y , y_had).
    
    Args:      
        eps (float): Small value to avoid division by zero
        g (list):    Segments length 
    
    Call:    
        x: input audio
        y: target audio
    
    
    Returns:
        average sum of cosine similary of a input and 
        target tensors sliced with various lengths
    """
    
    def __init__(self,
                eps = 1e-5,
                g = [508, 1016, 2032, 4062]):
        
        super(CosSimLoss, self).__init__()
        self.eps = eps
        self.g = g
        self.m = len(self.g)

    def cos_sim_func(self, input, target):
        '''
        Computes cosine similarity of input and target
        '''
        cos = nn.CosineSimilarity(dim=1, eps = self.eps)
        loss = 1 - cos(input, target)
        return loss
    
    
    def forward(self, x, y):
        
        loss = []
        for i in range(len(self.g)):     
          if i == 0:
            input =  x[:, : self.g[i]]
            target = y[:, : self.g[i]]
          
          else:
            input =  x[:, (self.g[i - 1]): self.g[i]]
            target = y[:, (self.g[i - 1]): self.g[i]]
          
          #calculate cosine similarity function
          c = self.cos_sim_func(input, target)
          loss.append(c)      
        
        return (1 / self.m * torch.sum(torch.FloatTensor(loss))), loss
