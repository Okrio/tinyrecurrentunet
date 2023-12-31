import time
import torch
import json
import argparse
import os
import onnxruntime as rt

from network import TRUNet


# load torch model
def load_model(model_path, network_config):
    model = TRUNet(**network_config)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

#torch inference function
def torch_inference(torch_model):
    x = torch.randn((1, 4, 257), dtype = torch.float32, requires_grad = False)
    start_time = time.time()
    with torch.no_grad():
      y = torch_model(x)
 
    end_time = time.time()
    return  end_time - start_time 


#onnx inference function
def onnx_inference(onnx_model_path):
    x = torch.randn(751, 4 , 257, dtype = torch.float32)
    ort_session = rt.InferenceSession(onnx_model_path)
    ort_input = {ort_session.get_inputs()[0].name: to_numpy(x)}
    
    #inference
    start_time = time.time()
    ort_outs = ort_session.run(None, ort_input)
    end_time = time.time()
    
    return (end_time - start_time)


def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()   
  

def time_average(lst):
  return sum(lst) / len(lst)
  

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, help = 'Path to config Json file')
  parser.add_argument('-i', '--ckpt_path', type=str, help = 'Path to trained torch model')
  parser.add_argument('-x', '--onnx_path', type=str, help = 'Path to onnx model')
  args = parser.parse_args()

  #load json file
  with open(args.config) as f:
    data = f.read()
  config = json.loads(data)
  
  #get device
  if torch.cuda.is_available():
        device = 'GPU' 
  else: 
    device = 'CPU'
  
  #load models
  model_config = config["network"]
  torch_model = load_model(args.ckpt_path,
                    model_config)
  onnx_model = args.onnx_path
  time_steps = 751 
  
  #list to save inference time in each in time
  time_keeper_torch = []
  for i in range(time_steps): 
    torch_time = torch_inference(torch_model)
    time_keeper_torch.append(torch_time)
   
  avg_torch_time = round(time_average(time_keeper_torch), 5)
  avg_onnx_time = round((onnx_inference(onnx_model) / time_steps), 5)
  
  print("Average inference times on {} for 2 seconds of audio:".format(device))
  print(f"Torch: {avg_torch_time} ms / {avg_torch_time * 1000} s")
  print(f"ONNX:  {avg_onnx_time} ms / {avg_onnx_time * 1000} s")
