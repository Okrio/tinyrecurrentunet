import time
import torch
import json
import argparse
import os

from network import TRUNet


# load torch model
def load_model(model_path, network_config):
    model = TRUNet(**network_config)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model
  
  
  
def time_average(lst):
  return sum(lst) / len(lst)
  

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, help = 'Path to config Json file')
  parser.add_argument('-i', '--ckpt_path', type=str, help = 'Path to trained torch model')
  args = parser.parse_args()

  #load json file
  with open(args.config) as f:
    data = f.read()
  config = json.loads(data)
  
  #load model
  model_config = config["network"]
  model = load_model(args.ckpt_path,
                    model_config)
  
  #create dummy input
  
  time_keeper = []
  
  for i in range(751):
    x = torch.randn((1, 4, 257),  requires_grad = False)
    start_time = time.time()
    with torch.no_grad():
      y = model(x)
 
    end_time = time.time()
    inf_time = (end_time - start_time)
    time_keeper.append(inf_time)
   
  avg_inf_time = time_average(time_keeper)
  print(f"Average inference time in ms: {avg_inf_time}")
