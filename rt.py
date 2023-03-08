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
      y = model(x)
 
    end_time = time.time()
    return =  end_time - start_time 


#onnx inference function
def onnx_inference(onnx_model_path):
    x = torch.randn(1, 4 , 257, dtype = torch.float32)
    ort_session = rt.InferenceSession(onnx_model_path)
    ort_input = {ort_session.get_inputs()[0].name: to_numpy(x)}
    
    #inference
    start_time = time.time()
    ort_outs = ort_session.run(None, ort_input)
    end_time = time.time()
    
    return (end_time - start_time)
    
  
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
  
  #load models
  model_config = config["network"]
  torch_model = load_model(args.ckpt_path,
                    model_config)
  onnx_model = args.onnx_path
  
  
  #list to save inference time in each in time
  time_keeper_torch = []
  time_keeper_onnx = []
  
  for i in range(751): 
    torch_time = torch_inference(torch_model)
    onnx_time = onnx_inference(onnx_model)
    
    time_keeper.append(torch_time)
    time_keeper.append(onnx_time)
   
  avg_torch_time = time_average(time_keeper_torch)
  avg_onnx_time = time_average(time_keeper_onnx)
  
  print("Average inference times in ms:")
  print(f"Torch: {avg_inf_time}")
  print(f"ONNX:  {avg_inf_time}")
